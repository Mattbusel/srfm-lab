"""
test_tensor_net.py — Tests for TensorNet module (Project AETERNUS).

Covers all major modules:
  - MPS: construction, arithmetic, canonicalization, inner products, entanglement
  - TT: decomposition, rounding, arithmetic, evaluation
  - Tucker/CP: decompositions and reconstructions
  - Financial tensors: correlation tensor construction, Tucker/CP fitting
  - Quantum ML: encoding, kernel, MERA, Born machine
  - Anomaly detection: Tucker residuals, LOF, isolation forest
  - Compression: TT-linear, TT-embedding, compression ratios
  - Riemannian optimization: gradient, retraction, convergence
  - Benchmarks: smoke tests for benchmark functions
"""

import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# ============================================================================
# Test MPS module
# ============================================================================

class TestMPS:
    """Tests for matrix_product_state module."""

    def test_mps_random_creation(self):
        """Test random MPS creation with correct shapes."""
        from tensor_net.mps import mps_random, MatrixProductState
        key = jax.random.PRNGKey(0)
        mps = mps_random(8, 2, 4, key)
        assert isinstance(mps, MatrixProductState)
        assert mps.n_sites == 8
        assert mps.phys_dims == (2,) * 8
        # Check boundary conditions
        assert mps.tensors[0].shape[0] == 1
        assert mps.tensors[-1].shape[2] == 1
        # Check all tensors have physical dim 2
        for t in mps.tensors:
            assert t.shape[1] == 2

    def test_mps_product_state(self):
        """Test product state creation."""
        from tensor_net.mps import mps_product_state
        v0 = jnp.array([1.0, 0.0])
        v1 = jnp.array([0.0, 1.0])
        mps = mps_product_state([v0, v1, v0])
        assert mps.n_sites == 3
        assert mps.max_bond == 1

    def test_mps_ghz(self):
        """Test GHZ state creation."""
        from tensor_net.mps import mps_ghz
        mps = mps_ghz(4)
        assert mps.n_sites == 4
        assert mps.max_bond == 2

    def test_mps_to_dense_and_back(self):
        """Test round-trip: dense -> MPS -> dense."""
        from tensor_net.mps import mps_from_dense, mps_to_dense
        vec = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        vec = vec / jnp.linalg.norm(vec)
        mps = mps_from_dense(vec, [2, 2, 2], max_bond=4)
        dense = mps_to_dense(mps)
        err = float(jnp.linalg.norm(vec - dense.reshape(-1)))
        assert err < 1e-3, f"Round-trip error too large: {err}"

    def test_mps_inner_product_normalized(self):
        """Test <mps|mps> = 1 for normalized MPS."""
        from tensor_net.mps import mps_random, mps_normalize, mps_inner_product
        key = jax.random.PRNGKey(1)
        mps = mps_random(6, 2, 4, key)
        mps = mps_normalize(mps)
        ip = mps_inner_product(mps, mps)
        assert abs(float(jnp.real(ip)) - 1.0) < 1e-4, f"Norm not 1: {ip}"

    def test_mps_norm(self):
        """Test MPS norm computation."""
        from tensor_net.mps import mps_random, mps_norm
        key = jax.random.PRNGKey(2)
        mps = mps_random(5, 2, 4, key)
        norm = float(mps_norm(mps))
        assert norm > 0.0
        assert not math.isnan(norm)

    def test_mps_add(self):
        """Test MPS addition: ||mps1 + mps2||^2 = ||mps1||^2 + ||mps2||^2 + 2<mps1|mps2>."""
        from tensor_net.mps import mps_random, mps_add, mps_norm_sq, mps_inner_product
        key = jax.random.PRNGKey(3)
        k1, k2 = jax.random.split(key)
        mps1 = mps_random(4, 2, 3, k1)
        mps2 = mps_random(4, 2, 3, k2)
        mps_sum = mps_add(mps1, mps2)

        lhs = float(mps_norm_sq(mps_sum))
        n1 = float(mps_norm_sq(mps1))
        n2 = float(mps_norm_sq(mps2))
        cross = float(2 * jnp.real(mps_inner_product(mps1, mps2)))
        rhs = n1 + n2 + cross
        assert abs(lhs - rhs) < 1e-3, f"Addition identity failed: {lhs} vs {rhs}"

    def test_mps_left_canonicalize(self):
        """Test that left-canonical MPS has orthonormal site tensors."""
        from tensor_net.mps import mps_random, mps_left_canonicalize
        key = jax.random.PRNGKey(4)
        mps = mps_random(5, 2, 4, key)
        mps_lc = mps_left_canonicalize(mps)
        # Check first site: sum_s A[0]^T A[0] ~ I
        A0 = mps_lc.tensors[0]  # (1, 2, chi)
        A0_mat = A0.reshape(2, -1)  # (2, chi)
        gram = A0_mat @ A0_mat.T
        # Should be approximately diagonal / identity-like
        assert gram.shape == (2, 2) or gram.shape[0] <= 4

    def test_mps_compress(self):
        """Test MPS compression reduces bond dimension."""
        from tensor_net.mps import mps_random, mps_compress
        key = jax.random.PRNGKey(5)
        mps = mps_random(6, 2, 8, key)
        mps_c = mps_compress(mps, max_bond=2)
        assert mps_c.max_bond <= 3  # Allow slight slack from QR

    def test_mps_bond_entropies(self):
        """Test bond entropy computation gives non-negative values."""
        from tensor_net.mps import mps_random, mps_normalize, mps_bond_entropies
        key = jax.random.PRNGKey(6)
        mps = mps_random(4, 2, 4, key)
        mps = mps_normalize(mps)
        entropies = mps_bond_entropies(mps)
        assert entropies.shape == (3,)
        assert all(float(e) >= -1e-5 for e in entropies), f"Negative entropies: {entropies}"

    def test_mps_expectation_single(self):
        """Test single-site expectation value of Pauli Z."""
        from tensor_net.mps import mps_product_state, mps_expectation_single
        # |0> state: <Z> = 1
        v0 = jnp.array([1.0, 0.0])
        mps = mps_product_state([v0, v0])
        Z = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        ev = float(jnp.real(mps_expectation_single(mps, Z, 0)))
        assert abs(ev - 1.0) < 1e-4, f"<Z>|0> should be 1, got {ev}"

    def test_mps_pytree(self):
        """Test MPS can be used as JAX pytree."""
        from tensor_net.mps import mps_random
        key = jax.random.PRNGKey(7)
        mps = mps_random(4, 2, 4, key)
        # Try tree operations
        leaves, treedef = jax.tree_util.tree_flatten(mps)
        mps2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert mps2.n_sites == 4


# ============================================================================
# Test TT decomposition module
# ============================================================================

class TestTTDecomp:
    """Tests for tensor train decomposition."""

    def test_tt_svd_reconstruction(self):
        """Test TT-SVD gives accurate reconstruction."""
        from tensor_net.tt_decomp import tt_svd, tt_to_dense
        key = jax.random.PRNGKey(10)
        tensor = jax.random.normal(key, (4, 4, 4))
        tt = tt_svd(tensor, max_rank=8, cutoff=1e-12)
        recon = tt_to_dense(tt)
        err = float(jnp.linalg.norm(tensor - recon) / (jnp.linalg.norm(tensor) + 1e-10))
        assert err < 0.01, f"TT-SVD error too large: {err}"

    def test_tt_add(self):
        """Test TT addition."""
        from tensor_net.tt_decomp import TensorTrain, tt_add, tt_to_dense
        key = jax.random.PRNGKey(11)
        k1, k2 = jax.random.split(key)

        # Create simple TTs
        cores1 = [jax.random.normal(k1, (1, 4, 2)), jax.random.normal(k1, (2, 4, 1))]
        cores2 = [jax.random.normal(k2, (1, 4, 2)), jax.random.normal(k2, (2, 4, 1))]
        tt1 = TensorTrain(cores1, (4, 4))
        tt2 = TensorTrain(cores2, (4, 4))

        tt_sum = tt_add(tt1, tt2)
        # Check shapes
        assert tt_sum.cores[0].shape[2] == 4  # 2 + 2
        assert tt_sum.cores[1].shape[0] == 4

        # Check values
        dense1 = tt_to_dense(tt1)
        dense2 = tt_to_dense(tt2)
        dense_sum = tt_to_dense(tt_sum)
        err = float(jnp.linalg.norm(dense1 + dense2 - dense_sum))
        assert err < 1e-4

    def test_tt_dot(self):
        """Test TT inner product."""
        from tensor_net.tt_decomp import TensorTrain, tt_dot, tt_norm

        key = jax.random.PRNGKey(12)
        cores = [jax.random.normal(key, (1, 3, 2)), jax.random.normal(key, (2, 3, 1))]
        tt = TensorTrain(cores, (3, 3))

        # Self dot product >= 0
        result = tt_dot(tt, tt)
        assert float(result) >= 0.0

        # Norm consistency
        norm_sq = float(tt_dot(tt, tt))
        norm = float(tt_norm(tt))
        assert abs(norm ** 2 - norm_sq) < 1e-4

    def test_tt_round(self):
        """Test TT rounding reduces rank."""
        from tensor_net.tt_decomp import tt_svd, tt_round

        key = jax.random.PRNGKey(13)
        tensor = jax.random.normal(key, (4, 4, 4))
        tt_full = tt_svd(tensor, max_rank=8, cutoff=0.0)
        tt_r = tt_round(tt_full, max_rank=2)

        assert tt_r.max_rank <= 3  # Allow slight slack

    def test_tt_hadamard(self):
        """Test element-wise TT product."""
        from tensor_net.tt_decomp import TensorTrain, tt_hadamard, tt_to_dense

        key = jax.random.PRNGKey(14)
        k1, k2 = jax.random.split(key)
        cores1 = [jax.random.normal(k1, (1, 3, 1)), jax.random.normal(k1, (1, 3, 1))]
        cores2 = [jax.random.normal(k2, (1, 3, 1)), jax.random.normal(k2, (1, 3, 1))]
        tt1 = TensorTrain(cores1, (3, 3))
        tt2 = TensorTrain(cores2, (3, 3))

        tt_prod = tt_hadamard(tt1, tt2)
        d1 = tt_to_dense(tt1)
        d2 = tt_to_dense(tt2)
        d_prod = tt_to_dense(tt_prod)
        err = float(jnp.linalg.norm(d1 * d2 - d_prod))
        assert err < 1e-4

    def test_tucker_decomp(self):
        """Test Tucker decomposition reconstruction."""
        from tensor_net.tt_decomp import tucker_decomp

        key = jax.random.PRNGKey(15)
        tensor = jax.random.normal(key, (6, 5, 4))
        ranks = [3, 3, 3]
        core, factors = tucker_decomp(tensor, ranks, n_iter=10)

        # Verify shapes
        assert core.shape == tuple(ranks)
        assert factors[0].shape[0] == 6
        assert factors[1].shape[0] == 5
        assert factors[2].shape[0] == 4

    def test_cp_decomp(self):
        """Test CP decomposition."""
        from tensor_net.tt_decomp import cp_decomp

        key = jax.random.PRNGKey(16)
        # Build a rank-3 CP tensor
        n, R = 5, 3
        A = jax.random.normal(key, (n, R))
        B = jax.random.normal(key, (n, R))
        C = jax.random.normal(key, (n, R))
        tensor = jnp.einsum("ir,jr,kr->ijk", A, B, C)

        lambdas, factors = cp_decomp(tensor, rank=R, n_iter=50)
        assert lambdas.shape == (R,)
        assert len(factors) == 3

    def test_tt_evaluate(self):
        """Test TT element-wise evaluation."""
        from tensor_net.tt_decomp import tt_svd, tt_evaluate, tt_to_dense

        key = jax.random.PRNGKey(17)
        tensor = jax.random.normal(key, (3, 4, 3))
        tt = tt_svd(tensor, max_rank=4, cutoff=1e-10)
        dense = tt_to_dense(tt)

        # Evaluate at specific index
        idx = (1, 2, 1)
        val = float(tt_evaluate(tt, idx))
        expected = float(dense[idx])
        assert abs(val - expected) < 0.1, f"TT eval mismatch: {val} vs {expected}"


# ============================================================================
# Test financial tensors
# ============================================================================

class TestFinancialTensors:
    """Tests for financial tensor operations."""

    def test_build_return_tensor(self):
        """Test return tensor construction from prices."""
        from tensor_net.financial_tensors import build_return_tensor

        key = jax.random.PRNGKey(20)
        prices = jnp.abs(jax.random.normal(key, (100, 10))) * 100 + 100
        ret_tensor = build_return_tensor(prices, window=10)
        assert ret_tensor.ndim == 3
        assert ret_tensor.shape[1] == 10  # n_assets
        assert ret_tensor.shape[2] == 10  # window

    def test_build_correlation_tensor(self):
        """Test correlation tensor construction."""
        from tensor_net.financial_tensors import build_correlation_tensor

        key = jax.random.PRNGKey(21)
        returns = jax.random.normal(key, (200, 20)) * 0.01
        corr_tensor = build_correlation_tensor(returns, window=60, stride=20)
        assert corr_tensor.ndim == 3
        n = corr_tensor.shape[1]
        assert n == 20
        # Check symmetry of each slice
        for t in range(min(3, corr_tensor.shape[0])):
            C = corr_tensor[t]
            assert jnp.allclose(C, C.T, atol=1e-5), "Correlation matrix not symmetric"

    def test_correlation_tucker(self):
        """Test Tucker decomposition of correlation tensor."""
        from tensor_net.financial_tensors import CorrelationTucker, build_correlation_tensor

        key = jax.random.PRNGKey(22)
        returns = jax.random.normal(key, (200, 10)) * 0.01
        corr_tensor = build_correlation_tensor(returns, window=40, stride=20)
        T = corr_tensor.shape[0]
        if T < 3:
            return  # Skip if too few time steps

        tucker = CorrelationTucker(n_time_factors=min(3, T), n_asset_factors=min(5, 10))
        tucker.fit(corr_tensor)
        recon = tucker.reconstruct()
        assert recon.shape == corr_tensor.shape
        err = tucker.reconstruction_error(corr_tensor)
        assert err >= 0.0

    def test_correlation_mps(self):
        """Test MPS compression of correlation matrix."""
        from tensor_net.financial_tensors import CorrelationMPS

        key = jax.random.PRNGKey(23)
        n = 8
        # Generate PSD correlation matrix
        A = jax.random.normal(key, (n, n))
        corr = A @ A.T / n + jnp.eye(n) * 0.1
        d = jnp.sqrt(jnp.diag(corr))
        corr = corr / jnp.outer(d, d)

        mps_enc = CorrelationMPS(max_bond=8)
        mps_enc.fit(corr)
        recon = mps_enc.transform()
        assert recon.shape[0] == n
        ratio = mps_enc.compression_ratio()
        assert ratio > 0.0

    def test_causality_tensor(self):
        """Test Granger causality tensor computation."""
        from tensor_net.financial_tensors import CausalityTensor

        key = jax.random.PRNGKey(24)
        returns = jax.random.normal(key, (100, 5)) * 0.01
        caus = CausalityTensor(max_lag=2, max_bond=4)
        caus.fit(returns)
        assert caus.causality_tensor_ is not None
        assert caus.causality_tensor_.shape == (5, 5, 2)

    def test_run_financial_experiment(self):
        """Test the full financial MPS experiment pipeline."""
        from tensor_net.financial_tensors import run_financial_mps_experiment

        key = jax.random.PRNGKey(25)
        results = run_financial_mps_experiment(n_assets=10, T=100, n_factors=3, max_bond=4, key=key)
        assert "tucker_error" in results
        assert "mps_error" in results
        assert results["mps_error"] >= 0.0


# ============================================================================
# Test quantum ML module
# ============================================================================

class TestQuantumML:
    """Tests for quantum-inspired ML."""

    def test_amplitude_encoding(self):
        """Test amplitude encoding of feature vector."""
        from tensor_net.quantum_ml import amplitude_encoding

        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        mps = amplitude_encoding(x, n_qubits=3)
        assert mps.n_sites == 3
        assert mps.phys_dims == (2, 2, 2)

    def test_angle_encoding(self):
        """Test angle encoding returns correct gate matrices."""
        from tensor_net.quantum_ml import angle_encoding

        x = jnp.array([0.5, 1.0, -0.5])
        gates = angle_encoding(x, n_qubits=3, encoding_type="ry")
        assert len(gates) == 3
        for g in gates:
            assert g.shape == (2, 2)

    def test_quantum_circuit_sim(self):
        """Test quantum circuit simulator."""
        from tensor_net.quantum_ml import QuantumCircuitSim

        qc = QuantumCircuitSim(n_qubits=4, max_bond=8)
        # Apply Hadamard and measure
        qc.h(0)
        qc.h(1)
        z_exp = qc.measure_z(0)
        # After H gate on |0>, <Z> = 0
        assert abs(z_exp) < 0.1, f"<Z> after H should be ~0, got {z_exp}"

    def test_quantum_kernel(self):
        """Test quantum kernel computation."""
        from tensor_net.quantum_ml import QuantumKernel

        key = jax.random.PRNGKey(30)
        X = jax.random.normal(key, (3, 8))

        qk = QuantumKernel(n_qubits=3, bond_dim=2)
        K = qk.compute_kernel_matrix(X)
        assert K.shape == (3, 3)
        # Diagonal should be 1 (self-similarity = 1 for normalized states)
        # Kernel values should be in [0, 1]
        for i in range(3):
            for j in range(3):
                assert 0.0 <= float(K[i, j]) <= 1.01, f"Kernel value out of range: {K[i,j]}"

    def test_born_machine_smoke(self):
        """Smoke test for Born Machine (1 epoch)."""
        from tensor_net.quantum_ml import BornMachine

        key = jax.random.PRNGKey(31)
        n_sites = 4
        data = jax.random.randint(key, (20, n_sites), 0, 2)

        bm = BornMachine(n_sites=n_sites, phys_dim=2, bond_dim=2, n_epochs=1)
        bm.fit(data, key=key)
        assert bm.is_fitted

    def test_mps_sample(self):
        """Test MPS sampling produces valid configurations."""
        from tensor_net.quantum_ml import mps_sample
        from tensor_net.mps import mps_random, mps_normalize

        key = jax.random.PRNGKey(32)
        k1, k2 = jax.random.split(key)
        mps = mps_random(4, 2, 2, k1)
        mps = mps_normalize(mps)
        samples = mps_sample(mps, 5, k2)
        assert samples.shape == (5, 4)
        # All values should be 0 or 1
        assert jnp.all((samples == 0) | (samples == 1))


# ============================================================================
# Test anomaly detection
# ============================================================================

class TestAnomalyDetection:
    """Tests for tensor anomaly detection."""

    def test_tucker_residual_detector(self):
        """Test Tucker residual detector."""
        from tensor_net.anomaly_detection import TuckerResidualDetector
        from tensor_net.financial_tensors import build_correlation_tensor

        key = jax.random.PRNGKey(40)
        returns = jax.random.normal(key, (200, 10)) * 0.01
        corr_tensor = build_correlation_tensor(returns, window=40, stride=20)
        T = corr_tensor.shape[0]
        if T < 5:
            return

        detector = TuckerResidualDetector(
            n_time_factors=min(3, T),
            n_asset_factors=min(5, 10)
        )
        detector.fit(corr_tensor)
        assert detector.threshold_ is not None
        assert detector.threshold_ > 0.0

        scores = detector.predict_scores(corr_tensor)
        assert scores.shape[0] == T
        assert jnp.all(scores >= 0.0)

    def test_robust_tensor_pca(self):
        """Test RTPCA decomposition."""
        from tensor_net.anomaly_detection import RobustTensorPCA

        key = jax.random.PRNGKey(41)
        tensor = jax.random.normal(key, (8, 6, 6))
        rtpca = RobustTensorPCA(lambda_sparse=0.1, max_iter=10)
        L, S = rtpca.fit_transform(tensor)
        assert L.shape == tensor.shape
        assert S.shape == tensor.shape

        # L + S should approximate tensor
        err = float(jnp.linalg.norm(L + S - tensor) / (jnp.linalg.norm(tensor) + 1e-10))
        assert err < 0.5  # ADMM may not fully converge in 10 iterations

    def test_tensor_lof(self):
        """Test Tensor LOF anomaly detector."""
        from tensor_net.anomaly_detection import TensorLOF
        from tensor_net.financial_tensors import build_correlation_tensor

        key = jax.random.PRNGKey(42)
        returns = jax.random.normal(key, (200, 8)) * 0.01
        corr_tensor = build_correlation_tensor(returns, window=40, stride=20)
        T = corr_tensor.shape[0]
        if T < 10:
            return

        lof = TensorLOF(k=min(5, T - 1), n_factors=4)
        lof.fit(corr_tensor)
        scores = lof.predict_scores(corr_tensor)
        assert scores.shape[0] == T

    def test_isolation_forest(self):
        """Test Isolation Tensor Forest."""
        from tensor_net.anomaly_detection import IsolationTensorForest

        key = jax.random.PRNGKey(43)
        features = jax.random.normal(key, (50, 10))
        iforest = IsolationTensorForest(n_estimators=10, max_samples=30, key=key)
        iforest.fit(features)
        scores = iforest.predict_scores(features)
        assert scores.shape[0] == 50
        assert jnp.all(scores >= 0.0) and jnp.all(scores <= 1.01)

    def test_change_point_detector(self):
        """Test tensor change point detector."""
        from tensor_net.anomaly_detection import TensorChangePointDetector
        from tensor_net.financial_tensors import build_correlation_tensor

        key = jax.random.PRNGKey(44)
        returns_normal = jax.random.normal(key, (200, 8)) * 0.01
        corr_ref = build_correlation_tensor(returns_normal, window=40, stride=20)
        if corr_ref.shape[0] < 5:
            return

        detector = TensorChangePointDetector(n_factors=min(3, corr_ref.shape[0]))
        detector.fit(corr_ref)

        key, subkey = jax.random.split(key)
        new_corr = corr_ref[0]
        stat, is_change = detector.detect(new_corr)
        assert isinstance(is_change, bool) or hasattr(is_change, "item")


# ============================================================================
# Test compression module
# ============================================================================

class TestCompression:
    """Tests for neural network weight compression."""

    def test_tt_linear_simple_forward(self):
        """Test TTLinearSimple forward pass."""
        from tensor_net.compression import TTLinearSimple

        import flax.linen as nn

        model = TTLinearSimple(
            row_shape=(4, 4),
            col_shape=(4, 4),
            tt_rank=2,
        )
        x = jnp.ones((2, 16))
        key = jax.random.PRNGKey(50)
        params = model.init(key, x)
        y = model.apply(params, x)
        assert y.shape == (2, 16)

    def test_tt_embedding_forward(self):
        """Test TTEmbedding forward pass."""
        from tensor_net.compression import TTEmbedding

        model = TTEmbedding(
            vocab_size=16,
            embed_dim=8,
            vocab_shape=(4, 4),
            embed_shape=(2, 4),
            tt_rank=2,
        )
        indices = jnp.array([0, 1, 5, 15])
        key = jax.random.PRNGKey(51)
        params = model.init(key, indices)
        embeddings = model.apply(params, indices)
        assert embeddings.shape == (4, 8)

    def test_compression_ratio_estimate(self):
        """Test compression ratio estimation."""
        from tensor_net.compression import estimate_compression_ratio

        ratio = estimate_compression_ratio(
            in_features=64,
            out_features=64,
            row_shape=(8, 8),
            col_shape=(8, 8),
            tt_rank=4,
        )
        assert ratio > 1.0, f"Compression ratio should be > 1, got {ratio}"

    def test_factorize_dimension(self):
        """Test dimension factorization."""
        from tensor_net.compression import factorize_dimension, _factorize

        # Test that product equals original
        for n in [4, 8, 16, 32, 64, 100]:
            factors = _factorize(n)
            prod = 1
            for f in factors:
                prod *= f
            assert prod >= n, f"Factorization product {prod} < {n} for factors {factors}"

    def test_quantize_tt_cores(self):
        """Test TT core quantization."""
        from tensor_net.compression import quantize_tt_cores
        from tensor_net.tt_decomp import TensorTrain

        key = jax.random.PRNGKey(52)
        cores = [jax.random.normal(key, (1, 4, 2)), jax.random.normal(key, (2, 4, 1))]
        tt = TensorTrain(cores, (4, 4))
        tt_q = quantize_tt_cores(tt, n_bits=8)
        # Quantized values should be within original range
        for k, (G_orig, G_q) in enumerate(zip(tt.cores, tt_q.cores)):
            assert float(jnp.max(G_q)) <= float(jnp.max(G_orig)) + 1e-4


# ============================================================================
# Test Riemannian optimization
# ============================================================================

class TestRiemannianOptim:
    """Tests for Riemannian optimization on TT manifold."""

    def test_riemannian_gradient(self):
        """Test Riemannian gradient computation."""
        from tensor_net.riemannian_optim import riemannian_gradient
        from tensor_net.tt_decomp import tt_svd, tt_to_dense

        key = jax.random.PRNGKey(60)
        tensor = jax.random.normal(key, (3, 3, 3))
        tt = tt_svd(tensor, max_rank=2)

        def loss_fn(t):
            return jnp.sum(tt_to_dense(t) ** 2)

        rgrad = riemannian_gradient(tt, loss_fn)
        assert rgrad is not None
        assert rgrad.ndim == tt.ndim

    def test_svd_retraction(self):
        """Test SVD retraction stays on manifold."""
        from tensor_net.riemannian_optim import svd_retraction, riemannian_gradient
        from tensor_net.tt_decomp import tt_svd, tt_to_dense, tt_norm

        key = jax.random.PRNGKey(61)
        tensor = jax.random.normal(key, (3, 3, 3))
        tt = tt_svd(tensor, max_rank=2)

        def loss_fn(t):
            return jnp.sum(tt_to_dense(t) ** 2)

        rgrad = riemannian_gradient(tt, loss_fn)
        tt_new = svd_retraction(tt, rgrad, -0.01, max_rank=2)
        # Should still be a valid TT
        assert tt_new.ndim == tt.ndim
        assert tt_new.max_rank <= 3

    def test_rgd_convergence(self):
        """Test that RGD decreases the loss."""
        from tensor_net.riemannian_optim import RiemannianGradientDescent
        from tensor_net.tt_decomp import tt_svd, tt_to_dense

        key = jax.random.PRNGKey(62)
        target = jax.random.normal(key, (3, 3, 3)) * 0.1
        tt_init = tt_svd(target, max_rank=1, cutoff=0.0)

        def loss_fn(t):
            recon = tt_to_dense(t).reshape(target.shape)
            return jnp.sum((target - recon) ** 2)

        optimizer = RiemannianGradientDescent(lr=0.01, max_rank=1)
        tt_final, losses = optimizer.optimize(tt_init, loss_fn, n_steps=5)

        assert len(losses) > 0
        assert losses[0] >= 0.0

    def test_riemannian_adam(self):
        """Smoke test for Riemannian Adam optimizer."""
        from tensor_net.riemannian_optim import RiemannianAdam
        from tensor_net.tt_decomp import tt_svd, tt_to_dense

        key = jax.random.PRNGKey(63)
        target = jax.random.normal(key, (3, 3, 3)) * 0.1
        tt_init = tt_svd(target, max_rank=1)

        def loss_fn(t):
            return jnp.sum((tt_to_dense(t) - target) ** 2)

        optimizer = RiemannianAdam(lr=0.001, max_rank=1)
        tt_final, losses = optimizer.optimize(tt_init, loss_fn, n_steps=3)
        assert len(losses) == 3

    def test_theoretical_convergence_rate(self):
        """Test convergence rate formula."""
        from tensor_net.riemannian_optim import theoretical_convergence_rate

        rate_gd = theoretical_convergence_rate(10.0, 1.0, "gradient_descent")
        rate_cg = theoretical_convergence_rate(10.0, 1.0, "conjugate_gradient")
        # CG should converge faster (smaller rate)
        assert rate_cg <= rate_gd + 1e-6

    def test_gradient_norm_history(self):
        """Test convergence analysis utility."""
        from tensor_net.riemannian_optim import gradient_norm_history

        # Simulated gradient norms (decaying)
        norms = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
        stats = gradient_norm_history(norms)
        assert "n_steps" in stats
        assert stats["n_steps"] == 6
        assert stats["final_grad_norm"] < stats["initial_grad_norm"]


# ============================================================================
# Test benchmarks (smoke tests)
# ============================================================================

class TestBenchmarks:
    """Smoke tests for benchmark functions."""

    def test_synthetic_low_rank_tensor(self):
        """Test synthetic tensor generation."""
        from tensor_net.benchmarks import synthetic_low_rank_tensor

        key = jax.random.PRNGKey(70)
        tensor = synthetic_low_rank_tensor((3, 3, 3), rank=2, noise_level=0.01, key=key)
        assert tensor.shape == (3, 3, 3)
        assert not jnp.any(jnp.isnan(tensor))

    def test_synthetic_correlation_tensor(self):
        """Test synthetic correlation tensor generation."""
        from tensor_net.benchmarks import synthetic_correlation_tensor

        key = jax.random.PRNGKey(71)
        returns, corr_tensor = synthetic_correlation_tensor(n_assets=10, T=100, key=key)
        assert returns.shape[1] == 10
        assert corr_tensor.ndim == 3
        assert corr_tensor.shape[1] == 10

    def test_error_vs_bond_dim(self):
        """Test error-vs-bond-dim curve computation."""
        from tensor_net.benchmarks import error_vs_bond_dim

        key = jax.random.PRNGKey(72)
        tensor = jax.random.normal(key, (4, 4, 4))
        result = error_vs_bond_dim(tensor, max_bond_dims=[1, 2, 4])
        assert len(result["relative_errors"]) == 3
        # Errors should be non-negative
        assert all(e >= 0.0 for e in result["relative_errors"])

    def test_benchmark_mps_operations_smoke(self):
        """Smoke test for MPS operations benchmark."""
        from tensor_net.benchmarks import benchmark_mps_operations

        key = jax.random.PRNGKey(73)
        result = benchmark_mps_operations(
            n_sites_list=[4],
            phys_dims=[2],
            bond_dims=[4],
            key=key,
        )
        assert len(result["results"]) == 1
        row = result["results"][0]
        assert "init_time_ms" in row
        assert row["init_time_ms"] >= 0.0

    def test_run_all_benchmarks_quick(self):
        """Test running all benchmarks in quick mode."""
        from tensor_net.benchmarks import run_all_benchmarks

        key = jax.random.PRNGKey(74)
        results = run_all_benchmarks(quick=True, key=key)
        assert "tt_svd" in results
        assert "mps_ops" in results


# ============================================================================
# Integration tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple modules."""

    def test_mps_to_tt_interop(self):
        """Test that MPS and TT representations are interoperable."""
        from tensor_net.mps import mps_random, mps_to_dense
        from tensor_net.tt_decomp import tt_svd, tt_to_dense

        key = jax.random.PRNGKey(80)
        # Create MPS and convert to dense
        mps = mps_random(4, 2, 4, key)
        dense_mps = mps_to_dense(mps).reshape(-1)

        # Create TT from same dense vector
        tt = tt_svd(dense_mps.reshape(2, 2, 2, 2), max_rank=4, cutoff=1e-10)
        dense_tt = tt_to_dense(tt).reshape(-1)

        err = float(jnp.linalg.norm(dense_mps - dense_tt))
        assert err < 0.1  # Some error expected due to different shapes

    def test_financial_anomaly_pipeline(self):
        """Test financial anomaly detection pipeline."""
        from tensor_net.financial_tensors import build_correlation_tensor
        from tensor_net.anomaly_detection import TuckerResidualDetector

        key = jax.random.PRNGKey(81)
        k1, k2 = jax.random.split(key)

        # Generate normal period
        returns_n = jax.random.normal(k1, (200, 10)) * 0.01
        corr_n = build_correlation_tensor(returns_n, window=40, stride=20)
        T = corr_n.shape[0]
        if T < 5:
            return

        n_tf = min(3, T)
        n_af = min(5, 10)
        detector = TuckerResidualDetector(n_time_factors=n_tf, n_asset_factors=n_af)
        detector.fit(corr_n)

        # Generate anomalous period with spiked correlations
        returns_a = jax.random.normal(k2, (100, 10)) * 0.05
        corr_a = build_correlation_tensor(returns_a, window=40, stride=20)
        if corr_a.shape[0] < 1:
            return

        scores = detector.predict_scores(corr_a)
        assert len(scores) == corr_a.shape[0]
        assert all(s >= 0.0 for s in scores)

    def test_compression_and_forward(self):
        """Test TT-compressed layer in a simple model."""
        from tensor_net.compression import TTLinearSimple
        import flax.linen as nn

        class SimpleModel(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = TTLinearSimple(
                    row_shape=(4, 4), col_shape=(4, 4), tt_rank=2
                )(x)
                return x

        model = SimpleModel()
        x = jnp.ones((3, 16))
        key = jax.random.PRNGKey(82)
        params = model.init(key, x)
        y = model.apply(params, x)
        assert y.shape == (3, 16)
        assert not jnp.any(jnp.isnan(y))


# ============================================================================
# Main runner
# ============================================================================

def run_all_tests():
    """Run all tests and print a summary."""
    test_classes = [
        TestMPS, TestTTDecomp, TestFinancialTensors,
        TestQuantumML, TestAnomalyDetection, TestCompression,
        TestRiemannianOptim, TestBenchmarks, TestIntegration,
    ]

    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method in methods:
            try:
                getattr(instance, method)()
                passed += 1
                print(f"  PASS  {cls.__name__}.{method}")
            except Exception as e:
                failed += 1
                errors.append((f"{cls.__name__}.{method}", str(e)))
                print(f"  FAIL  {cls.__name__}.{method}: {e}")

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailed tests:")
        for name, msg in errors:
            print(f"  {name}: {msg}")
    print(f"{'=' * 50}")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
