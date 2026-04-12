"""
TensorNet — Tensor Network / Matrix Product State framework for SRFM-lab Project AETERNUS.

Provides JAX-based MPS/TT decompositions and quantum-inspired methods for compressing
high-dimensional financial correlation structures.

Version: 0.1.0
"""

__version__ = "0.1.0"
__description__ = (
    "Tensor Network / Matrix Product State framework for compressing "
    "high-dimensional financial correlation structures (Project AETERNUS Module 3)."
)
__author__ = "SRFM-lab"

from tensor_net.mps import (
    MatrixProductState,
    mps_inner_product,
    mps_compress,
    mps_left_canonicalize,
    mps_right_canonicalize,
    mps_add,
    mps_scale,
    mps_expectation_single,
    mps_expectation_two_site,
    mps_to_density_matrix,
    mps_bond_entropies,
    mps_from_dense,
    dmrg_fit,
)

from tensor_net.tensor_train import (
    TensorTrain,
    tt_svd,
    tt_cross,
    tt_round,
    tt_add,
    tt_hadamard,
    tt_dot,
    tt_norm,
    tt_relative_error,
    TensorTrainMatrix,
    tt_matvec,
    tt_riemannian_grad,
)

from tensor_net.financial_compression import (
    CorrelationMPS,
    CausalityTensor,
    DependencyHypercube,
    StreamingCompressor,
    RegimeCompression,
    AnomalyDetector,
    run_financial_mps_experiment,
)

from tensor_net.quantum_inspired import (
    QuantumCircuitSim,
    VariationalPortfolioOptimizer,
    QuantumKernel,
    MERA,
    mps_sample,
    BornMachine,
)

from tensor_net.training import (
    TensorNetTrainer,
    reconstruction_mse,
    negative_log_likelihood,
    cross_entropy_loss,
    riemannian_sgd_step,
    cosine_schedule,
)

from tensor_net.visualization import (
    plot_bond_dimensions,
    plot_entanglement_spectrum,
    plot_compression_error_vs_ratio,
    plot_anomaly_scores,
    plot_tt_structure,
    animate_mps_evolution,
    plot_quantum_kernel_matrix,
)

from tensor_net.experiments import (
    experiment_correlation_compression,
    experiment_crisis_anomaly_detection,
    experiment_quantum_kernel_regime,
    experiment_causal_tensor_compression,
    experiment_portfolio_vqe,
)

__all__ = [
    # MPS
    "MatrixProductState", "mps_inner_product", "mps_compress",
    "mps_left_canonicalize", "mps_right_canonicalize", "mps_add", "mps_scale",
    "mps_expectation_single", "mps_expectation_two_site", "mps_to_density_matrix",
    "mps_bond_entropies", "mps_from_dense", "dmrg_fit",
    # TT
    "TensorTrain", "tt_svd", "tt_cross", "tt_round", "tt_add", "tt_hadamard",
    "tt_dot", "tt_norm", "tt_relative_error", "TensorTrainMatrix", "tt_matvec",
    "tt_riemannian_grad",
    # Financial
    "CorrelationMPS", "CausalityTensor", "DependencyHypercube",
    "StreamingCompressor", "RegimeCompression", "AnomalyDetector",
    "run_financial_mps_experiment",
    # Quantum
    "QuantumCircuitSim", "VariationalPortfolioOptimizer", "QuantumKernel",
    "MERA", "mps_sample", "BornMachine",
    # Training
    "TensorNetTrainer", "reconstruction_mse", "negative_log_likelihood",
    "cross_entropy_loss", "riemannian_sgd_step", "cosine_schedule",
    # Visualization
    "plot_bond_dimensions", "plot_entanglement_spectrum",
    "plot_compression_error_vs_ratio", "plot_anomaly_scores",
    "plot_tt_structure", "animate_mps_evolution", "plot_quantum_kernel_matrix",
    # Experiments
    "experiment_correlation_compression", "experiment_crisis_anomaly_detection",
    "experiment_quantum_kernel_regime", "experiment_causal_tensor_compression",
    "experiment_portfolio_vqe",
]
