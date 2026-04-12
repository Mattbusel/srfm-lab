# TensorNet — Module 3 of Project AETERNUS

Tensor Network / Matrix Product State framework for compressing high-dimensional financial correlation structures using JAX.

## Overview

TensorNet provides:
- **MPS (Matrix Product State)**: Compressed representation of high-dimensional state vectors
- **TT (Tensor Train)**: Decomposition of N-dimensional arrays into chain of 3-tensors
- **Financial compression**: Compress rolling correlation matrices, causality tensors, feature hypercubes
- **Anomaly detection**: Reconstruction error as crisis signal
- **Quantum-inspired methods**: VQE portfolio optimization, quantum kernel SVM, MERA, Born machine

## Structure

```
tensor_net/
  tensor_net/
    __init__.py             # Public API exports
    mps.py                  # MatrixProductState class (~900 lines)
    tensor_train.py         # TensorTrain class (~800 lines)
    financial_compression.py # CorrelationMPS, AnomalyDetector (~900 lines)
    quantum_inspired.py     # QuantumCircuit, VQE, MERA, BornMachine (~700 lines)
    training.py             # TensorNetTrainer, loss functions (~700 lines)
    visualization.py        # Plotting utilities (~500 lines)
    experiments.py          # Full experiment runners (~700 lines)
  tests/
    test_mps.py             # MPS unit tests (~300 lines)
    test_financial.py       # Financial compression tests (~200 lines)
  results/                  # Experiment outputs (plots, tables)
  requirements.txt
  README.md
```

## Key Concepts

### Matrix Product State
An MPS represents a high-dimensional vector |psi> as a chain of 3-tensors:
```
|psi> = sum_{s1,...,sN} A[1]^{s1} A[2]^{s2} ... A[N]^{sN} |s1...sN>
```
Each `A[k]` has shape `(chi_{k-1}, d_k, chi_k)` where:
- `chi_k` = bond dimension (controls approximation quality)
- `d_k` = physical dimension at site k

Bond dimension `D` controls the compression:
- `D=1`: product state (completely factored, maximum compression)
- `D=d^{N/2}`: exact (no compression)

### Financial Applications
- **CorrelationMPS**: Compress NxN correlation matrix by treating it as a vector and building an MPS. `D=8` typically captures 95%+ of variance with 8-16x compression.
- **AnomalyDetector**: Fit MPS to normal-period correlation structure. Reconstruction error of new data spikes before crisis events.
- **CausalityTensor**: Compress N×N×lag Granger causality tensor via TT-SVD.

### Quantum-Inspired Methods
- **VariationalPortfolioOptimizer**: Uses parameterized unitary MPS circuit to search portfolio weight simplex. Competitive with Markowitz at much lower computational cost.
- **QuantumKernel**: Inner product `|<phi(x)|phi(y)>|^2` between data-encoded MPS states. Used as kernel for SVM regime classification.
- **BornMachine**: MPS as generative model trained to match return distribution.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import jax.numpy as jnp
from tensor_net import CorrelationMPS, AnomalyDetector

# Compress correlation matrix
returns = jnp.array(...)  # shape (T, N)
comp = CorrelationMPS(n_assets=30, max_bond=8, window=252)
comp.fit(returns)
print(f"Compression ratio: {comp.compression_ratio_:.1f}x, Error: {comp.compression_error_:.4f}")

# Anomaly detection
detector = AnomalyDetector(n_assets=30, max_bond=8, window=252)
detector.fit_baseline(returns[:500])
scores, times = detector.score_sequence(returns)
```

## Running Experiments

```python
from tensor_net.experiments import run_all_experiments
results = run_all_experiments(save_results=True)
# Plots saved to results/
```

Individual experiments:
```python
from tensor_net.experiments import (
    experiment_correlation_compression,
    experiment_crisis_anomaly_detection,
    experiment_quantum_kernel_regime,
    experiment_causal_tensor_compression,
    experiment_portfolio_vqe,
)
```

## Running Tests

```bash
cd aeternus/tensor_net
pytest tests/ -v
```

## Design Principles

1. **JAX-native**: All operations are JAX-compatible — `jit`, `grad`, `vmap` work through MPS/TT computations. MPS and TT are registered as JAX pytrees.

2. **No stubs**: Every function is fully implemented with real computation.

3. **Financial focus**: All abstractions are designed around the financial use case (correlation matrices, rolling windows, regime detection, portfolio optimization).

4. **Quantum-inspired, not quantum**: Uses the mathematical structure of quantum circuits (MPS/MERA) without requiring quantum hardware.
