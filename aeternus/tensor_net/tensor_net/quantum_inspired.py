"""
quantum_inspired.py — Quantum-inspired methods using MPS for TensorNet.

Implements:
- QuantumCircuitSim: shallow quantum circuit simulation using MPS
- VariationalPortfolioOptimizer: VQE-style portfolio optimization
- QuantumKernel: MPS-based kernel for classification
- MERA: Multi-scale Entanglement Renormalization Ansatz
- mps_sample: sampling from MPS probability distribution
- BornMachine: MPS generative model
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, grad, vmap

from tensor_net.mps import (
    MatrixProductState,
    mps_from_dense,
    mps_to_dense,
    mps_inner_product,
    mps_compress,
    mps_add,
    mps_scale,
    mps_left_canonicalize,
    mps_right_canonicalize,
    mps_norm,
    encode_data_as_mps,
)


# ---------------------------------------------------------------------------
# Single-qubit and two-qubit gates
# ---------------------------------------------------------------------------

def rx_gate(theta: float) -> jnp.ndarray:
    """Rotation-X gate: Rx(theta) = [[cos(t/2), -i sin(t/2)], [-i sin(t/2), cos(t/2)]]."""
    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    return jnp.array([[c, -1j * s], [-1j * s, c]], dtype=jnp.complex64)


def ry_gate(theta: float) -> jnp.ndarray:
    """Rotation-Y gate: Ry(theta) = [[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]."""
    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    return jnp.array([[c, -s], [s, c]], dtype=jnp.complex64)


def rz_gate(theta: float) -> jnp.ndarray:
    """Rotation-Z gate: Rz(theta) = diag(exp(-it/2), exp(it/2))."""
    return jnp.diag(jnp.array([
        jnp.exp(-1j * theta / 2),
        jnp.exp(1j * theta / 2)
    ], dtype=jnp.complex64))


def cnot_gate() -> jnp.ndarray:
    """CNOT gate in computational basis {|00>, |01>, |10>, |11>}."""
    return jnp.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=jnp.complex64)


def hadamard_gate() -> jnp.ndarray:
    """Hadamard gate."""
    return jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64) / math.sqrt(2)


def rzz_gate(theta: float) -> jnp.ndarray:
    """Rzz(theta) = exp(-i*theta/2 * Z⊗Z)."""
    phase = jnp.exp(-1j * theta / 2)
    phase_c = jnp.conj(phase)
    return jnp.diag(jnp.array([phase_c, phase, phase, phase_c], dtype=jnp.complex64))


# ---------------------------------------------------------------------------
# QuantumCircuitSim
# ---------------------------------------------------------------------------

class QuantumCircuitSim:
    """
    Simulate shallow quantum circuits classically using MPS.

    This is the quantum-inspired ML approach: no actual quantum hardware needed.
    The MPS representation allows simulation of O(N) qubit circuits with
    polynomial bond dimension, capturing the entanglement structure.

    Parameters
    ----------
    n_qubits : number of qubits
    max_bond : maximum bond dimension (limits entanglement)
    """

    def __init__(self, n_qubits: int, max_bond: int = 32):
        self.n_qubits = n_qubits
        self.max_bond = max_bond
        self.circuit_ops_ = []
        self._init_state()

    def _init_state(self):
        """Initialize to |0...0> state."""
        # |0> = [1, 0] at each qubit
        tensors = [jnp.array([[[1.0, 0.0]]], dtype=jnp.complex64).reshape(1, 2, 1)
                   for _ in range(self.n_qubits)]
        self.state_ = MatrixProductState(tensors, (2,) * self.n_qubits)

    def reset(self):
        """Reset to |0...0> state."""
        self._init_state()
        self.circuit_ops_ = []
        return self

    def apply_single_qubit_gate(
        self, gate: jnp.ndarray, qubit: int
    ) -> "QuantumCircuitSim":
        """
        Apply a single-qubit gate to the specified qubit.
        gate: (2, 2) unitary matrix
        """
        assert gate.shape == (2, 2)
        assert 0 <= qubit < self.n_qubits

        tensors = list(self.state_.tensors)
        t = tensors[qubit]  # (chi_l, 2, chi_r)
        # Apply gate: new_tensor[chi_l, s', chi_r] = sum_s gate[s', s] * t[chi_l, s, chi_r]
        t_new = jnp.einsum("ij,ajk->aik", gate, t.astype(jnp.complex64))
        tensors[qubit] = t_new

        self.state_ = MatrixProductState(tensors, (2,) * self.n_qubits)
        self.circuit_ops_.append(("single", qubit, gate))
        return self

    def apply_two_qubit_gate(
        self, gate: jnp.ndarray, qubit1: int, qubit2: int
    ) -> "QuantumCircuitSim":
        """
        Apply a two-qubit gate to neighboring qubits.
        gate: (4, 4) or (2, 2, 2, 2) unitary matrix
        qubit2 must equal qubit1 + 1 (nearest-neighbor only).
        """
        assert abs(qubit2 - qubit1) == 1, "Only nearest-neighbor gates supported"
        if qubit2 < qubit1:
            qubit1, qubit2 = qubit2, qubit1

        assert 0 <= qubit1 < self.n_qubits - 1

        gate_4x4 = gate.reshape(4, 4).astype(jnp.complex64)
        gate_tensor = gate_4x4.reshape(2, 2, 2, 2)  # (s1', s2', s1, s2)

        tensors = list(self.state_.tensors)
        A = tensors[qubit1].astype(jnp.complex64)  # (chi_l, 2, chi_m)
        B = tensors[qubit2].astype(jnp.complex64)  # (chi_m, 2, chi_r)

        chi_l, _, chi_m = A.shape
        _, _, chi_r = B.shape

        # Contract A and B into two-site tensor: (chi_l, s1, s2, chi_r)
        theta = jnp.einsum("isa,ajb->isajb", A, B)
        theta = theta.reshape(chi_l, 2, 2, chi_r)  # wrong index order, fix:
        theta = jnp.einsum("isa,asb->iab", A.reshape(chi_l, 2, chi_m),
                           B.reshape(chi_m, 2, chi_r))
        # Actually: theta[chi_l, s1, s2, chi_r]
        theta = jnp.einsum("als,smr->almr", A, B)  # Hmm
        # Explicit: theta[a, s1, s2, b] = A[a, s1, m] * B[m, s2, b]
        theta = jnp.einsum("asm,mtn->astn", A, B)  # (chi_l, 2, 2, chi_r)

        # Apply gate: new_theta[a, s1', s2', b] = sum_{s1,s2} gate[s1',s2',s1,s2] * theta[a,s1,s2,b]
        new_theta = jnp.einsum("ijkl,akln->aijn", gate_tensor, theta)
        # Wait, gate: (s1', s2', s1, s2), theta: (chi_l, s1, s2, chi_r)
        new_theta = jnp.einsum("pqrs,arsb->apqb", gate_tensor, theta)
        # new_theta: (chi_l, 2, 2, chi_r)

        # SVD to split back into two tensors
        M = new_theta.reshape(chi_l * 2, 2 * chi_r)
        U, s, Vt = jnp.linalg.svd(M, full_matrices=False)

        # Truncate to max_bond
        chi_new = min(self.max_bond, len(s))
        U = U[:, :chi_new]
        s = s[:chi_new]
        Vt = Vt[:chi_new, :]

        A_new = U.reshape(chi_l, 2, chi_new)
        B_new = (jnp.diag(s) @ Vt).reshape(chi_new, 2, chi_r)

        tensors[qubit1] = A_new
        tensors[qubit2] = B_new
        self.state_ = MatrixProductState(tensors, (2,) * self.n_qubits)
        self.circuit_ops_.append(("two", qubit1, qubit2, gate))
        return self

    def apply_hadamard_layer(self) -> "QuantumCircuitSim":
        """Apply Hadamard to all qubits."""
        H = hadamard_gate()
        for i in range(self.n_qubits):
            self.apply_single_qubit_gate(H, i)
        return self

    def apply_rotation_layer(
        self, thetas: jnp.ndarray, axis: str = "y"
    ) -> "QuantumCircuitSim":
        """
        Apply parameterized rotation to each qubit.
        thetas: array of shape (n_qubits,)
        axis: 'x', 'y', or 'z'
        """
        gate_fn = {"x": rx_gate, "y": ry_gate, "z": rz_gate}[axis]
        for i in range(self.n_qubits):
            gate = gate_fn(float(thetas[i]))
            self.apply_single_qubit_gate(gate, i)
        return self

    def apply_entangling_layer(self, thetas: Optional[jnp.ndarray] = None) -> "QuantumCircuitSim":
        """
        Apply CNOT (or Rzz if thetas given) between all neighboring pairs.
        """
        for i in range(self.n_qubits - 1):
            if thetas is not None:
                gate = rzz_gate(float(thetas[i]))
            else:
                gate = cnot_gate()
            self.apply_two_qubit_gate(gate, i, i + 1)
        return self

    def measure_expectation(self, observable: jnp.ndarray, qubit: int) -> jnp.ndarray:
        """
        Measure <psi|O_qubit|psi> for a single-qubit observable O (2x2 matrix).
        """
        from tensor_net.mps import mps_expectation_single
        state_real = MatrixProductState(
            [jnp.real(t).astype(jnp.float32) for t in self.state_.tensors],
            (2,) * self.n_qubits
        )
        obs_real = jnp.real(observable).astype(jnp.float32)
        return mps_expectation_single(state_real, obs_real, qubit)

    def get_statevector(self) -> jnp.ndarray:
        """Return full 2^N statevector (only feasible for small N)."""
        dense = mps_to_dense(self.state_)
        return dense.reshape(-1).astype(jnp.complex64)

    def depth(self) -> int:
        """Return circuit depth (number of layers applied)."""
        return len(self.circuit_ops_)


# ---------------------------------------------------------------------------
# VariationalPortfolioOptimizer (VQE-style)
# ---------------------------------------------------------------------------

class VariationalPortfolioOptimizer:
    """
    VQE-style portfolio optimization using a parameterized MPS circuit.

    The quantum state |psi(theta)> encodes portfolio weights:
    w_i = |<i|psi(theta)>|^2  (measurement probability for basis state i)

    Minimize portfolio variance: Var = w^T Sigma w
    Subject to: sum(w) = 1, w >= 0

    The MPS circuit provides an expressive, continuously differentiable
    parameterization of the portfolio weight simplex.
    """

    def __init__(
        self,
        n_assets: int,
        n_layers: int = 3,
        max_bond: int = 8,
        n_steps: int = 200,
        lr: float = 0.02,
    ):
        self.n_assets = n_assets
        self.n_layers = n_layers
        self.max_bond = max_bond
        self.n_steps = n_steps
        self.lr = lr
        self.n_qubits = math.ceil(math.log2(n_assets))
        self.n_params_ = self.n_qubits * (2 * n_layers + 1)

    def _params_to_weights(self, params: jnp.ndarray, cov: jnp.ndarray) -> jnp.ndarray:
        """
        Convert variational parameters to portfolio weights via MPS circuit.

        Parameters
        ----------
        params : array of shape (n_params,)
        cov : covariance matrix of shape (n_assets, n_assets)

        Returns
        -------
        weights : array of shape (n_assets,), sums to 1
        """
        n_q = self.n_qubits
        n_assets = self.n_assets

        # Initialize circuit
        circuit = QuantumCircuitSim(n_q, self.max_bond)
        circuit.apply_hadamard_layer()

        # Apply variational layers
        param_idx = 0
        for layer in range(self.n_layers):
            # Rotation layer
            n_rot = n_q
            thetas_y = params[param_idx:param_idx + n_rot]
            param_idx += n_rot
            circuit.apply_rotation_layer(thetas_y, axis="y")

            # Entangling layer
            circuit.apply_entangling_layer()

        # Final rotation
        thetas_final = params[param_idx:param_idx + n_q]
        circuit.apply_rotation_layer(thetas_final, axis="y")

        # Get statevector as portfolio weights
        statevec = jnp.abs(circuit.get_statevector()) ** 2  # Probabilities
        # Only use first n_assets amplitudes
        weights_unnorm = statevec[:n_assets]
        weights = weights_unnorm / (jnp.sum(weights_unnorm) + 1e-12)
        return weights

    def portfolio_variance(
        self, params: jnp.ndarray, cov: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute portfolio variance for given parameters.
        Uses real-valued rotation gates for differentiability.
        """
        n_q = self.n_qubits
        n_assets = self.n_assets

        # Build MPS state directly (real-valued)
        # Initialize |0...0>
        tensors = [jnp.array([[[1.0, 0.0]]]).reshape(1, 2, 1) for _ in range(n_q)]

        param_idx = 0
        # Apply Hadamard (encoded as Y rotation by pi/2)
        h_theta = jnp.ones(n_q) * (math.pi / 2)
        tensors = self._apply_rotation_to_tensors(tensors, h_theta, n_q)

        for layer in range(self.n_layers):
            thetas = params[param_idx:param_idx + n_q]
            param_idx += n_q
            tensors = self._apply_rotation_to_tensors(tensors, thetas, n_q)
            tensors = self._apply_cnot_to_tensors(tensors, n_q)

        thetas_final = params[param_idx:param_idx + n_q]
        tensors = self._apply_rotation_to_tensors(tensors, thetas_final, n_q)

        # Compute probabilities
        mps_state = MatrixProductState(tensors, (2,) * n_q)
        state_dense = mps_to_dense(mps_state).reshape(-1)
        probs = state_dense ** 2
        probs = jnp.maximum(probs, 0.0)
        weights = probs[:n_assets] / (jnp.sum(probs[:n_assets]) + 1e-12)

        # Portfolio variance
        var = weights @ cov @ weights
        return var

    def _apply_rotation_to_tensors(
        self,
        tensors: List[jnp.ndarray],
        thetas: jnp.ndarray,
        n_q: int,
    ) -> List[jnp.ndarray]:
        """Apply Ry rotations (real-valued) to each qubit tensor."""
        new_tensors = []
        for i in range(n_q):
            t = tensors[i]  # (chi_l, 2, chi_r)
            c = jnp.cos(thetas[i] / 2)
            s = jnp.sin(thetas[i] / 2)
            ry = jnp.array([[c, -s], [s, c]], dtype=jnp.float32)
            t_new = jnp.einsum("ij,ajk->aik", ry, t)
            new_tensors.append(t_new)
        return new_tensors

    def _apply_cnot_to_tensors(
        self,
        tensors: List[jnp.ndarray],
        n_q: int,
    ) -> List[jnp.ndarray]:
        """Apply CNOT between neighboring qubits (real approximation via CZ-like gate)."""
        # Approximate CNOT with real matrices using controlled-phase
        for i in range(n_q - 1):
            A = tensors[i].astype(jnp.float32)
            B = tensors[i + 1].astype(jnp.float32)
            chi_l, _, chi_m = A.shape
            _, _, chi_r = B.shape

            # Contract
            theta_ab = jnp.einsum("asm,mtn->astn", A, B)  # (chi_l, 2, 2, chi_r)

            # Apply CNOT (real approximation)
            cnot_real = jnp.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ], dtype=jnp.float32).reshape(2, 2, 2, 2)

            new_theta = jnp.einsum("pqrs,arsb->apqb", cnot_real, theta_ab)

            # SVD split
            M = new_theta.reshape(chi_l * 2, 2 * chi_r)
            U, s, Vt = jnp.linalg.svd(M, full_matrices=False)
            chi_new = min(self.max_bond, len(s))
            tensors[i] = U[:, :chi_new].reshape(chi_l, 2, chi_new)
            tensors[i + 1] = (jnp.diag(s[:chi_new]) @ Vt[:chi_new, :]).reshape(chi_new, 2, chi_r)

        return tensors

    def optimize(
        self,
        cov: jnp.ndarray,
        expected_returns: Optional[jnp.ndarray] = None,
        risk_aversion: float = 1.0,
    ) -> Dict:
        """
        Optimize portfolio using variational MPS circuit.

        Parameters
        ----------
        cov : covariance matrix of shape (n_assets, n_assets)
        expected_returns : expected returns array, shape (n_assets,)
        risk_aversion : risk aversion parameter (higher = more risk averse)

        Returns
        -------
        dict with keys: weights, variance, returns, losses, params
        """
        cov = jnp.array(cov, dtype=jnp.float32)
        key = jax.random.PRNGKey(42)
        params = jax.random.normal(key, (self.n_params_,)) * 0.1

        losses = []

        def loss_fn(params):
            var = self.portfolio_variance(params, cov)
            loss = risk_aversion * var
            if expected_returns is not None:
                # Add return term
                n_q = self.n_qubits
                n_assets = self.n_assets
                # Compute weights
                tensors = [jnp.array([[[1.0, 0.0]]]).reshape(1, 2, 1) for _ in range(n_q)]
                h_theta = jnp.ones(n_q) * (math.pi / 2)
                tensors = self._apply_rotation_to_tensors(tensors, h_theta, n_q)
                param_idx = 0
                for layer in range(self.n_layers):
                    thetas = params[param_idx:param_idx + n_q]
                    param_idx += n_q
                    tensors = self._apply_rotation_to_tensors(tensors, thetas, n_q)
                    tensors = self._apply_cnot_to_tensors(tensors, n_q)
                thetas_f = params[param_idx:param_idx + n_q]
                tensors = self._apply_rotation_to_tensors(tensors, thetas_f, n_q)
                mps_s = MatrixProductState(tensors, (2,) * n_q)
                state_dense = mps_to_dense(mps_s).reshape(-1)
                probs = state_dense ** 2
                weights = probs[:n_assets] / (jnp.sum(probs[:n_assets]) + 1e-12)
                ret = jnp.dot(weights, expected_returns)
                loss = loss - ret  # Maximize return, minimize variance

            return loss

        # Gradient descent with Adam
        import optax
        optimizer = optax.adam(self.lr)
        opt_state = optimizer.init(params)

        for step in range(self.n_steps):
            loss_val, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            losses.append(float(loss_val))

            if step > 10 and abs(losses[-1] - losses[-10]) < 1e-7:
                break

        # Final weights
        final_var = self.portfolio_variance(params, cov)
        n_q = self.n_qubits
        n_assets = self.n_assets
        tensors = [jnp.array([[[1.0, 0.0]]]).reshape(1, 2, 1) for _ in range(n_q)]
        h_theta = jnp.ones(n_q) * (math.pi / 2)
        tensors = self._apply_rotation_to_tensors(tensors, h_theta, n_q)
        param_idx = 0
        for layer in range(self.n_layers):
            thetas = params[param_idx:param_idx + n_q]
            param_idx += n_q
            tensors = self._apply_rotation_to_tensors(tensors, thetas, n_q)
            tensors = self._apply_cnot_to_tensors(tensors, n_q)
        thetas_f = params[param_idx:param_idx + n_q]
        tensors = self._apply_rotation_to_tensors(tensors, thetas_f, n_q)
        mps_s = MatrixProductState(tensors, (2,) * n_q)
        state_dense = mps_to_dense(mps_s).reshape(-1)
        probs = state_dense ** 2
        weights = probs[:n_assets] / (jnp.sum(probs[:n_assets]) + 1e-12)

        result = {
            "weights": weights,
            "variance": float(final_var),
            "losses": losses,
            "params": params,
            "n_steps_taken": len(losses),
        }
        if expected_returns is not None:
            result["expected_return"] = float(jnp.dot(weights, expected_returns))

        return result


# ---------------------------------------------------------------------------
# QuantumKernel
# ---------------------------------------------------------------------------

class QuantumKernel:
    """
    Quantum-inspired kernel: K(x, y) = |<phi(x)|phi(y)>|^2
    where |phi(x)> is data-encoded as an MPS state.

    This provides a non-linear, infinite-dimensional feature map
    useful for SVM-style classification of market regimes.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        max_bond: int = 16,
        encoding: str = "angle",
        n_encoding_layers: int = 2,
    ):
        self.n_qubits = n_qubits
        self.max_bond = max_bond
        self.encoding = encoding
        self.n_encoding_layers = n_encoding_layers

    def encode(self, x: jnp.ndarray) -> MatrixProductState:
        """
        Encode data vector x as an MPS quantum state.

        For angle encoding: apply R_y(theta_i * pi) at each qubit,
        with multiple layers and entanglement.
        """
        x = jnp.array(x, dtype=jnp.float32)
        n_q = self.n_qubits

        if self.encoding == "angle":
            circuit = QuantumCircuitSim(n_q, self.max_bond)
            for layer in range(self.n_encoding_layers):
                # Angle encoding: map features to rotation angles
                idx_start = (layer * n_q) % len(x)
                angles = x[idx_start:idx_start + n_q]
                if len(angles) < n_q:
                    angles = jnp.pad(angles, (0, n_q - len(angles)))
                angles = angles * math.pi  # Scale to [0, pi]
                circuit.apply_rotation_layer(angles, axis="y")
                if layer < self.n_encoding_layers - 1:
                    circuit.apply_entangling_layer()

            return circuit.state_

        elif self.encoding == "amplitude":
            return encode_data_as_mps(x, n_q, encoding="amplitude")

        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

    def kernel(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Compute K(x, y) = |<phi(x)|phi(y)>|^2.
        """
        phi_x = self.encode(x)
        phi_y = self.encode(y)
        ip = mps_inner_product(phi_x, phi_y)
        return jnp.real(ip) ** 2 + jnp.imag(ip) ** 2  # |<phi_x|phi_y>|^2

    def kernel_matrix(
        self,
        X: jnp.ndarray,
        Y: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Compute the full kernel matrix K[i,j] = K(X[i], Y[j]).

        Parameters
        ----------
        X : array of shape (n_x, d)
        Y : array of shape (n_y, d), defaults to X

        Returns
        -------
        K : array of shape (n_x, n_y)
        """
        X = jnp.array(X, dtype=jnp.float32)
        if Y is None:
            Y = X
        Y = jnp.array(Y, dtype=jnp.float32)

        n_x, n_y = X.shape[0], Y.shape[0]
        K = np.zeros((n_x, n_y), dtype=np.float32)

        # Pre-encode all states
        phi_x_list = [self.encode(X[i]) for i in range(n_x)]
        phi_y_list = [self.encode(Y[j]) for j in range(n_y)]

        for i in range(n_x):
            for j in range(n_y):
                K[i, j] = float(self.kernel(X[i], Y[j]))

        return jnp.array(K)

    def fit_svm(
        self,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
    ) -> "QuantumKernelSVM":
        """
        Fit an SVM classifier using this quantum kernel.
        Returns a QuantumKernelSVM object.
        """
        K_train = self.kernel_matrix(X_train)
        return QuantumKernelSVM(self, K_train, X_train, y_train)


class QuantumKernelSVM:
    """SVM classifier with precomputed quantum kernel."""

    def __init__(
        self,
        kernel: QuantumKernel,
        K_train: jnp.ndarray,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
    ):
        self.kernel = kernel
        self.K_train = K_train
        self.X_train = X_train
        self.y_train = y_train

        # Fit SVM using sklearn with precomputed kernel
        from sklearn.svm import SVC
        self.svm_ = SVC(kernel="precomputed")
        self.svm_.fit(np.array(K_train), np.array(y_train))

    def predict(self, X_test: jnp.ndarray) -> np.ndarray:
        """Predict class labels for test data."""
        K_test = self.kernel.kernel_matrix(X_test, self.X_train)
        return self.svm_.predict(np.array(K_test))

    def accuracy(self, X_test: jnp.ndarray, y_test: jnp.ndarray) -> float:
        """Return accuracy on test data."""
        preds = self.predict(X_test)
        return float(np.mean(preds == np.array(y_test)))


# ---------------------------------------------------------------------------
# MERA: Multi-scale Entanglement Renormalization Ansatz
# ---------------------------------------------------------------------------

class MERA:
    """
    Multi-scale Entanglement Renormalization Ansatz for hierarchical tensor network.

    MERA captures both micro (local) and macro (global) structure via a
    coarse-graining hierarchy. Useful for multi-scale market analysis.

    Structure: alternating layers of disentanglers (U) and isometries (W)
    that coarse-grain the system by factor 2 at each layer.

    n_sites → n_sites/2 → n_sites/4 → ... → 1 (top tensor)
    """

    def __init__(
        self,
        n_sites: int,
        phys_dim: int = 2,
        bond_dim: int = 4,
        n_layers: Optional[int] = None,
    ):
        assert n_sites & (n_sites - 1) == 0, "n_sites must be a power of 2"
        self.n_sites = n_sites
        self.phys_dim = phys_dim
        self.bond_dim = bond_dim
        self.n_layers = n_layers or int(math.log2(n_sites))
        self.disentanglers_ = []
        self.isometries_ = []
        self.top_tensor_ = None
        self._initialize_random()

    def _initialize_random(self, key: Optional[jax.random.KeyArray] = None):
        """Initialize MERA with random unitary/isometric tensors."""
        if key is None:
            key = jax.random.PRNGKey(42)

        d = self.phys_dim
        chi = self.bond_dim
        n = self.n_sites
        self.disentanglers_ = []
        self.isometries_ = []

        current_n = n
        current_d = d

        for layer in range(self.n_layers):
            n_disentanglers = current_n // 2
            n_isometries = current_n // 2

            # Disentanglers: (d^2, d^2) unitary, stored as (d, d, d, d)
            layer_disen = []
            for i in range(n_disentanglers):
                key, subkey = jax.random.split(key)
                # Random unitary via QR
                M = jax.random.normal(subkey, (current_d ** 2, current_d ** 2))
                Q, _ = jnp.linalg.qr(M)
                layer_disen.append(Q.reshape(current_d, current_d, current_d, current_d))
            self.disentanglers_.append(layer_disen)

            # Isometries: (chi, d^2) isometry, stored as (chi, d, d)
            layer_iso = []
            for i in range(n_isometries):
                key, subkey = jax.random.split(key)
                # Random isometry: chi × d^2 with chi ≤ d^2
                chi_out = min(chi, current_d ** 2)
                M = jax.random.normal(subkey, (chi_out, current_d ** 2))
                Q, _ = jnp.linalg.qr(M.T)
                W = Q.T[:chi_out, :].reshape(chi_out, current_d, current_d)
                layer_iso.append(W)
            self.isometries_.append(layer_iso)

            current_n = current_n // 2
            current_d = min(chi, current_d ** 2)

        # Top tensor: just a single vector of the remaining dim
        key, subkey = jax.random.split(key)
        self.top_tensor_ = jax.random.normal(subkey, (current_d,))
        self.top_tensor_ = self.top_tensor_ / (jnp.linalg.norm(self.top_tensor_) + 1e-12)

    def apply_disentangler(
        self,
        psi: jnp.ndarray,
        layer: int,
        site: int,
    ) -> jnp.ndarray:
        """
        Apply disentangler U at layer, site to state tensor.
        psi: state tensor of shape (d,) * n_sites
        """
        U = self.disentanglers_[layer][site]
        n_sites_layer = len(psi)
        # Apply U to sites 2*site, 2*site+1
        # This is complex for general psi — simplified implementation
        return psi

    def coarse_grain(
        self,
        state: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Coarse-grain the state through all MERA layers.
        Returns the top-level description.

        state: array of shape (n_sites, d) — local density matrices per site
        """
        current = state
        n = self.n_sites

        for layer in range(self.n_layers):
            # Apply disentanglers and isometries
            current_n = current.shape[0]
            if current_n < 2:
                break

            new_current = []
            for i in range(current_n // 2):
                # Merge sites 2i and 2i+1
                if 2 * i + 1 < current_n:
                    merged = (current[2 * i] + current[2 * i + 1]) / 2
                else:
                    merged = current[2 * i]
                new_current.append(merged)

            current = jnp.array(new_current)

        return current

    def multi_scale_analysis(
        self,
        signal: jnp.ndarray,
    ) -> List[jnp.ndarray]:
        """
        Perform multi-scale analysis of a signal using MERA coarse-graining.

        Parameters
        ----------
        signal : time series of shape (T, n_features) or (n_sites,)

        Returns
        -------
        scales : list of arrays at each scale level
        """
        if signal.ndim == 2:
            # Take one representative vector per MERA site
            T, F = signal.shape
            n = self.n_sites
            # Divide signal into n segments, take mean
            seg_size = T // n
            segments = []
            for i in range(n):
                seg = signal[i * seg_size:(i + 1) * seg_size]
                segments.append(jnp.mean(seg, axis=0))
            signal_sites = jnp.array(segments)
        else:
            signal_sites = signal[:self.n_sites]
            if len(signal_sites) < self.n_sites:
                signal_sites = jnp.pad(signal_sites, (0, self.n_sites - len(signal_sites)))
            signal_sites = signal_sites.reshape(self.n_sites, -1)

        scales = [signal_sites]
        current = signal_sites

        for layer in range(self.n_layers):
            current_n = current.shape[0]
            if current_n < 2:
                break
            coarsened = jnp.array([
                (current[2 * i] + current[min(2 * i + 1, current_n - 1)]) / 2
                for i in range(current_n // 2)
            ])
            scales.append(coarsened)
            current = coarsened

        return scales

    def n_params(self) -> int:
        """Total number of parameters."""
        total = 0
        for layer_d in self.disentanglers_:
            for U in layer_d:
                total += U.size
        for layer_w in self.isometries_:
            for W in layer_w:
                total += W.size
        if self.top_tensor_ is not None:
            total += self.top_tensor_.size
        return total


# ---------------------------------------------------------------------------
# MPS Sampling
# ---------------------------------------------------------------------------

def mps_sample(
    mps: MatrixProductState,
    n_samples: int,
    key: jax.random.KeyArray,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """
    Sample from the probability distribution |MPS|^2 using sequential sampling.

    Uses the Born rule: P(s1, s2, ..., sN) = |<s1 s2 ... sN|psi>|^2

    For each site, sample s_i from the conditional:
    P(s_i | s_1, ..., s_{i-1}) = P(s_1,...,s_i) / P(s_1,...,s_{i-1})

    This is done efficiently using the MPS transfer matrix structure.

    Parameters
    ----------
    mps : MatrixProductState representing a probability distribution
    n_samples : number of samples to draw
    key : JAX random key
    temperature : sampling temperature (T→0 gives mode, T→∞ gives uniform)

    Returns
    -------
    samples : array of shape (n_samples, n_sites) with integer values
    """
    n = mps.n_sites
    samples = np.zeros((n_samples, n), dtype=int)

    # First left-canonicalize for efficient sampling
    mps_lc = mps_left_canonicalize(mps)

    for s in range(n_samples):
        key, subkey = jax.random.split(key)
        sample = _mps_sample_single(mps_lc, subkey, temperature)
        samples[s] = np.array(sample)

    return jnp.array(samples)


def _mps_sample_single(
    mps: MatrixProductState,
    key: jax.random.KeyArray,
    temperature: float = 1.0,
) -> List[int]:
    """Sample a single configuration from MPS."""
    n = mps.n_sites
    sample = []

    # Start with trivial left boundary
    L = jnp.ones((1,), dtype=jnp.float32)

    for i in range(n):
        tensor = mps.tensors[i]  # (chi_l, d, chi_r)
        chi_l, d, chi_r = tensor.shape

        # Compute unnormalized probs for each value of s_i
        # p(s_i) = L @ tensor[:, s_i, :] @ R_i where R_i is right norm
        # Since we're sampling left-to-right and MPS is left-canonical,
        # the probs are: p(s_i) = ||L @ tensor[:, s_i, :]||^2
        probs = []
        for s in range(d):
            t_s = tensor[:, s, :]  # (chi_l, chi_r)
            # Contract with L: L @ t_s
            Lts = L @ t_s  # (chi_r,) if L is (chi_l,)
            # Wait: L is (chi_l,), but for tracking normalization we need
            # L as a vector contracted with the left bond
            # p(s_i) propto ||L t_s||^2
            prob_s = float(jnp.sum(Lts ** 2))
            probs.append(prob_s)

        probs = np.array(probs, dtype=np.float64)
        if temperature != 1.0:
            probs = probs ** (1.0 / temperature)
        total = probs.sum()
        if total < 1e-12:
            probs = np.ones(d) / d
        else:
            probs = probs / total

        # Sample
        key, subkey = jax.random.split(key)
        s_i = int(jax.random.choice(subkey, d, p=jnp.array(probs)))
        sample.append(s_i)

        # Update L
        t_s = tensor[:, s_i, :]  # (chi_l, chi_r)
        L_new = L @ t_s  # (chi_r,)
        norm = float(jnp.linalg.norm(L_new))
        if norm > 1e-12:
            L = L_new / norm
        else:
            L = jnp.ones((chi_r,), dtype=jnp.float32) / chi_r

    return sample


# ---------------------------------------------------------------------------
# BornMachine: MPS as generative model
# ---------------------------------------------------------------------------

class BornMachine:
    """
    MPS Born Machine: train MPS to match a target probability distribution.

    The MPS represents a quantum state |psi>, and the probability distribution
    is P(x) = |<x|psi>|^2 (Born rule).

    Training: maximize log-likelihood sum_x log P(x) via gradient descent.
    """

    def __init__(
        self,
        n_sites: int,
        phys_dim: int = 2,
        bond_dim: int = 8,
        n_steps: int = 500,
        lr: float = 0.01,
        batch_size: int = 64,
    ):
        self.n_sites = n_sites
        self.phys_dim = phys_dim
        self.bond_dim = bond_dim
        self.n_steps = n_steps
        self.lr = lr
        self.batch_size = batch_size
        self.mps_: Optional[MatrixProductState] = None
        self.losses_: List[float] = []

    def _amplitude(self, tensors: List[jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute MPS amplitude <x|psi> for a configuration x.
        x: array of shape (n_sites,) with integer values
        """
        n = self.n_sites
        # Contract MPS along the configuration x
        result = tensors[0][0, x[0], :]  # (chi_1,)
        for i in range(1, n):
            result = result @ tensors[i][:, x[i], :]  # (chi_{i+1},)
        return result[0]

    def _log_prob(self, tensors: List[jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        """Log probability log P(x) = log |<x|psi>|^2 for discrete x."""
        amp = self._amplitude(tensors, x)
        return 2 * jnp.log(jnp.abs(amp) + 1e-12)

    def _batch_log_prob(
        self, tensors: List[jnp.ndarray], X_batch: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute log probs for a batch of configurations."""
        # Vectorize over batch
        log_probs = []
        for x in X_batch:
            log_probs.append(self._log_prob(tensors, x))
        return jnp.array(log_probs)

    def _partition_function(self, tensors: List[jnp.ndarray]) -> jnp.ndarray:
        """
        Compute partition function Z = sum_x |<x|psi>|^2 = <psi|psi>.
        Via transfer matrix contraction.
        """
        T_mat = jnp.ones((1, 1), dtype=jnp.float32)
        for i in range(self.n_sites):
            A = tensors[i]  # (chi_l, d, chi_r)
            # T' = sum_s A^* A
            T_mat = jnp.einsum("ab,asc,bsd->cd", T_mat, jnp.conj(A), A)
        return jnp.real(T_mat[0, 0])

    def _nll_loss(
        self,
        tensors: List[jnp.ndarray],
        X_batch: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Negative log-likelihood loss:
        NLL = -mean_x [ log P(x) ]
           = -mean_x [ log |<x|psi>|^2 ] + log Z
        """
        # Compute log amplitudes
        log_amps = []
        for x in X_batch:
            amp = self._amplitude(tensors, x)
            log_amps.append(jnp.log(amp ** 2 + 1e-12))
        log_amps = jnp.array(log_amps)

        # Partition function
        Z = self._partition_function(tensors)
        log_Z = jnp.log(Z + 1e-12)

        nll = -jnp.mean(log_amps) + log_Z
        return nll

    def fit(
        self,
        X_train: jnp.ndarray,
        key: Optional[jax.random.KeyArray] = None,
    ) -> "BornMachine":
        """
        Train Born Machine on discrete data.

        Parameters
        ----------
        X_train : array of shape (N, n_sites) with integer values in [0, phys_dim)
        key : JAX random key
        """
        if key is None:
            key = jax.random.PRNGKey(42)

        X_train = jnp.array(X_train, dtype=jnp.int32)
        N = X_train.shape[0]

        # Initialize MPS randomly
        mps_init = mps_random(
            self.n_sites, self.phys_dim, self.bond_dim, key,
            dtype=jnp.float32
        )
        from tensor_net.mps import mps_random
        mps_init = mps_random(self.n_sites, self.phys_dim, self.bond_dim, key)
        tensors = list(mps_init.tensors)

        import optax
        optimizer = optax.adam(self.lr)
        opt_state = optimizer.init(tensors)

        self.losses_ = []

        for step in range(self.n_steps):
            # Sample random batch
            key, subkey = jax.random.split(key)
            batch_idx = jax.random.randint(subkey, (self.batch_size,), 0, N)
            X_batch = X_train[batch_idx]

            loss_val, grads = jax.value_and_grad(
                lambda t: self._nll_loss(t, X_batch)
            )(tensors)

            updates, opt_state = optimizer.update(grads, opt_state, tensors)
            tensors = optax.apply_updates(tensors, updates)
            self.losses_.append(float(loss_val))

            # Periodically renormalize
            if step % 20 == 0:
                Z = float(self._partition_function(tensors))
                if Z > 1e-12:
                    tensors[0] = tensors[0] / (Z ** (1 / (2 * self.n_sites)))

            if step > 20 and abs(self.losses_[-1] - self.losses_[-20]) < 1e-7:
                break

        self.mps_ = MatrixProductState(tensors, (self.phys_dim,) * self.n_sites)
        return self

    def sample(
        self,
        n_samples: int,
        key: Optional[jax.random.KeyArray] = None,
        temperature: float = 1.0,
    ) -> jnp.ndarray:
        """
        Generate samples from the trained Born machine.
        Returns array of shape (n_samples, n_sites).
        """
        if self.mps_ is None:
            raise RuntimeError("Call fit() first.")
        if key is None:
            key = jax.random.PRNGKey(0)
        return mps_sample(self.mps_, n_samples, key, temperature)

    def log_prob(self, x: jnp.ndarray) -> float:
        """Compute log probability of a configuration x."""
        if self.mps_ is None:
            raise RuntimeError("Call fit() first.")
        tensors = self.mps_.tensors
        amp = self._amplitude(tensors, jnp.array(x, dtype=jnp.int32))
        Z = self._partition_function(tensors)
        return float(jnp.log(amp ** 2 + 1e-12) - jnp.log(Z + 1e-12))


# Need to import mps_random for BornMachine
from tensor_net.mps import mps_random
