"""
benchmarks.py — Benchmarks for TensorNet (Project AETERNUS).

Benchmarks cover:
  - Compression ratio vs. reconstruction accuracy (TT-SVD, Tucker, MPS)
  - Bond dimension vs. approximation error curves
  - Timing benchmarks: TT-SVD, MPS operations, contraction costs
  - Memory usage analysis
  - Financial tensor compression benchmarks (500 assets, 1000 time steps)
  - Quantum kernel evaluation timing
  - Anomaly detection precision-recall curves
  - Riemannian optimizer convergence comparison
  - Scalability: N-dimensional tensors, varying n_sites
  - Rank estimation accuracy
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple, Sequence, Dict, Any, Callable

import jax
import jax.numpy as jnp
import numpy as np


# ============================================================================
# Synthetic data generators
# ============================================================================

def synthetic_low_rank_tensor(
    shape: Tuple[int, ...],
    rank: int,
    noise_level: float = 0.01,
    key: jax.random.KeyArray = jax.random.PRNGKey(0),
) -> jnp.ndarray:
    """
    Generate a synthetic low-rank tensor for benchmarking.

    Parameters
    ----------
    shape : tensor shape
    rank : true TT-rank
    noise_level : noise fraction of signal norm
    key : random key

    Returns
    -------
    Noisy low-rank tensor
    """
    from .tt_decomp import TensorTrain, tt_to_dense

    n_dims = len(shape)
    key, *subkeys = jax.random.split(key, n_dims + 1)

    cores = []
    for k in range(n_dims):
        r_l = 1 if k == 0 else rank
        r_r = 1 if k == n_dims - 1 else rank
        G = jax.random.normal(subkeys[k], (r_l, shape[k], r_r))
        cores.append(G)

    tt = TensorTrain(cores, shape)
    signal = tt_to_dense(tt)

    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, shape)
    noise_scale = noise_level * jnp.linalg.norm(signal) / (jnp.linalg.norm(noise) + 1e-10)

    return signal + noise_scale * noise


def synthetic_correlation_tensor(
    n_assets: int = 100,
    T: int = 500,
    n_factors: int = 5,
    key: jax.random.KeyArray = jax.random.PRNGKey(42),
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate synthetic returns and correlation tensor.

    Parameters
    ----------
    n_assets : number of assets
    T : number of time steps
    n_factors : number of latent factors
    key : random key

    Returns
    -------
    (returns, correlation_tensor) where corr has shape (T//20, n_assets, n_assets)
    """
    from .financial_tensors import build_correlation_tensor

    key, k1, k2, k3 = jax.random.split(key, 4)
    factor_ret = jax.random.normal(k1, (T, n_factors)) * 0.01
    loadings = jax.random.normal(k2, (n_assets, n_factors)) * 0.5
    idio = jax.random.normal(k3, (T, n_assets)) * 0.005
    returns = factor_ret @ loadings.T + idio

    corr_tensor = build_correlation_tensor(returns, window=60, stride=20)
    return returns, corr_tensor


# ============================================================================
# Compression benchmarks
# ============================================================================

def benchmark_tt_svd_compression(
    shapes: List[Tuple[int, ...]],
    ranks: List[int],
    noise_level: float = 0.001,
    key: jax.random.KeyArray = jax.random.PRNGKey(0),
) -> Dict[str, Any]:
    """
    Benchmark TT-SVD compression across different tensor shapes and ranks.

    Parameters
    ----------
    shapes : list of tensor shapes to test
    ranks : list of TT-ranks to test
    noise_level : synthetic data noise level
    key : random key

    Returns
    -------
    Dictionary with benchmark results
    """
    from .tt_decomp import tt_svd, tt_to_dense, TensorTrain

    results = []

    for shape in shapes:
        for true_rank in ranks:
            key, subkey = jax.random.split(key)
            tensor = synthetic_low_rank_tensor(shape, true_rank, noise_level, subkey)

            # Try different compression ranks
            for comp_rank in [2, 4, 8, 16, 32]:
                if comp_rank > true_rank * 3:
                    continue

                t_start = time.time()
                tt = tt_svd(tensor, max_rank=comp_rank, cutoff=1e-10)
                t_svd = time.time() - t_start

                t_start = time.time()
                recon = tt_to_dense(tt)
                t_recon = time.time() - t_start

                err = float(jnp.linalg.norm(tensor - recon) / (jnp.linalg.norm(tensor) + 1e-10))
                n_dense = 1
                for s in shape:
                    n_dense *= s
                compression = n_dense / (tt.n_params + 1e-10)

                results.append({
                    "shape": shape,
                    "true_rank": true_rank,
                    "comp_rank": comp_rank,
                    "relative_error": err,
                    "compression_ratio": compression,
                    "tt_svd_time_s": t_svd,
                    "reconstruction_time_s": t_recon,
                    "n_params_dense": n_dense,
                    "n_params_tt": tt.n_params,
                })

    return {"results": results, "summary": _summarize_results(results)}


def benchmark_tucker_compression(
    corr_shapes: List[Tuple[int, int, int]],
    time_ranks: List[int],
    asset_ranks: List[int],
    key: jax.random.KeyArray = jax.random.PRNGKey(1),
) -> Dict[str, Any]:
    """
    Benchmark Tucker compression for correlation tensors.

    Parameters
    ----------
    corr_shapes : list of (T, n_assets, n_assets) shapes
    time_ranks : Tucker ranks for time dimension
    asset_ranks : Tucker ranks for asset dimensions
    key : random key

    Returns
    -------
    Benchmark results dictionary
    """
    from .tt_decomp import tucker_decomp
    from .financial_tensors import CorrelationTucker

    results = []

    for (T, n, _) in corr_shapes:
        key, subkey = jax.random.split(key)
        _, corr_tensor = synthetic_correlation_tensor(n, T, n_factors=5, key=subkey)
        T_corr = corr_tensor.shape[0]

        for t_rank in time_ranks:
            for a_rank in asset_ranks:
                t_rank_actual = min(t_rank, T_corr)
                a_rank_actual = min(a_rank, n)

                tucker = CorrelationTucker(
                    n_time_factors=t_rank_actual,
                    n_asset_factors=a_rank_actual,
                )

                t_start = time.time()
                tucker.fit(corr_tensor)
                t_fit = time.time() - t_start

                recon = tucker.reconstruct()
                err = float(jnp.linalg.norm(corr_tensor - recon) / (jnp.linalg.norm(corr_tensor) + 1e-10))

                n_dense = T_corr * n * n
                core_sz = tucker.core_.size if tucker.core_ is not None else 0
                factor_sz = sum(f.size for f in tucker.factors_) if tucker.factors_ else 0
                n_tucker = core_sz + factor_sz

                results.append({
                    "T_corr": T_corr,
                    "n_assets": n,
                    "time_rank": t_rank_actual,
                    "asset_rank": a_rank_actual,
                    "relative_error": err,
                    "compression_ratio": n_dense / (n_tucker + 1e-10),
                    "fit_time_s": t_fit,
                    "n_params_dense": n_dense,
                    "n_params_tucker": n_tucker,
                })

    return {"results": results, "summary": _summarize_results(results)}


def benchmark_mps_operations(
    n_sites_list: List[int] = [4, 8, 12, 16],
    phys_dims: List[int] = [2, 4],
    bond_dims: List[int] = [4, 8, 16],
    key: jax.random.KeyArray = jax.random.PRNGKey(2),
) -> Dict[str, Any]:
    """
    Benchmark core MPS operations.

    Tests: random init, inner product, compression, canonicalization.

    Parameters
    ----------
    n_sites_list : list of chain lengths
    phys_dims : physical dimensions
    bond_dims : bond dimensions
    key : random key

    Returns
    -------
    Timing results dictionary
    """
    from .mps import (
        mps_random, mps_inner_product, mps_compress,
        mps_left_canonicalize, mps_norm
    )

    results = []

    for n in n_sites_list:
        for d in phys_dims:
            for chi in bond_dims:
                key, sk1, sk2 = jax.random.split(key, 3)

                # Init
                t_start = time.time()
                mps1 = mps_random(n, d, chi, sk1)
                mps2 = mps_random(n, d, chi, sk2)
                t_init = time.time() - t_start

                # Inner product
                t_start = time.time()
                ip = mps_inner_product(mps1, mps2)
                t_ip = time.time() - t_start

                # Left canonicalization
                t_start = time.time()
                _ = mps_left_canonicalize(mps1)
                t_lc = time.time() - t_start

                # Compression
                t_start = time.time()
                _ = mps_compress(mps1, max_bond=chi // 2 + 1)
                t_comp = time.time() - t_start

                results.append({
                    "n_sites": n,
                    "phys_dim": d,
                    "bond_dim": chi,
                    "n_params": mps1.num_params(),
                    "init_time_ms": t_init * 1000,
                    "inner_product_time_ms": t_ip * 1000,
                    "canonicalization_time_ms": t_lc * 1000,
                    "compression_time_ms": t_comp * 1000,
                })

    return {"results": results}


# ============================================================================
# Bond dimension vs. error curves
# ============================================================================

def error_vs_bond_dim(
    tensor: jnp.ndarray,
    max_bond_dims: List[int],
) -> Dict[str, Any]:
    """
    Compute reconstruction error as a function of bond dimension.

    Parameters
    ----------
    tensor : input tensor to compress
    max_bond_dims : list of maximum bond dimensions to test

    Returns
    -------
    Dictionary with errors and compression ratios per bond dimension
    """
    from .tt_decomp import tt_svd, tt_to_dense

    n_dense = 1
    for s in tensor.shape:
        n_dense *= s

    errors = []
    ratios = []
    n_params = []

    for chi in max_bond_dims:
        tt = tt_svd(tensor, max_rank=chi, cutoff=1e-14)
        recon = tt_to_dense(tt)
        err = float(jnp.linalg.norm(tensor - recon) / (jnp.linalg.norm(tensor) + 1e-10))
        errors.append(err)
        ratios.append(n_dense / (tt.n_params + 1e-10))
        n_params.append(tt.n_params)

    return {
        "bond_dims": max_bond_dims,
        "relative_errors": errors,
        "compression_ratios": ratios,
        "n_params": n_params,
        "n_dense": n_dense,
        "tensor_shape": tensor.shape,
    }


def error_vs_tucker_rank(
    tensor: jnp.ndarray,
    ranks_range: List[int],
) -> Dict[str, Any]:
    """
    Tucker reconstruction error vs. rank.

    Parameters
    ----------
    tensor : 3D input tensor
    ranks_range : list of Tucker ranks to test

    Returns
    -------
    Dictionary with errors per rank
    """
    from .tt_decomp import tucker_decomp

    n_dense = 1
    for s in tensor.shape:
        n_dense *= s

    n_dims = tensor.ndim
    errors = []
    ratios = []

    for r in ranks_range:
        ranks = [min(r, tensor.shape[k]) for k in range(n_dims)]
        core, factors = tucker_decomp(tensor, ranks, n_iter=10)

        # Reconstruct
        recon = core
        for k in range(n_dims - 1, -1, -1):
            recon = jnp.tensordot(recon, factors[k], axes=([0], [1]))
            recon = jnp.moveaxis(recon, -1, 0)

        err = float(jnp.linalg.norm(tensor - recon) / (jnp.linalg.norm(tensor) + 1e-10))
        n_tucker = core.size + sum(f.size for f in factors)
        errors.append(err)
        ratios.append(n_dense / (n_tucker + 1e-10))

    return {
        "ranks": ranks_range,
        "relative_errors": errors,
        "compression_ratios": ratios,
        "tensor_shape": tensor.shape,
    }


# ============================================================================
# Scalability benchmarks
# ============================================================================

def scalability_benchmark(
    n_assets_list: List[int] = [10, 50, 100, 200, 500],
    T: int = 500,
    max_bond: int = 16,
    key: jax.random.KeyArray = jax.random.PRNGKey(5),
) -> Dict[str, Any]:
    """
    Benchmark TensorNet scalability with number of assets.

    Parameters
    ----------
    n_assets_list : list of asset counts to benchmark
    T : number of time steps
    max_bond : MPS bond dimension
    key : random key

    Returns
    -------
    Scalability results
    """
    from .financial_tensors import CorrelationMPS
    from .anomaly_detection import TuckerResidualDetector

    results = []

    for n_assets in n_assets_list:
        key, subkey = jax.random.split(key)
        returns, corr_tensor = synthetic_correlation_tensor(n_assets, T, key=subkey)
        T_corr = corr_tensor.shape[0]

        # MPS compression timing
        t_start = time.time()
        mps_enc = CorrelationMPS(max_bond=max_bond)
        mps_enc.fit(corr_tensor[0])
        t_mps = time.time() - t_start

        # Tucker anomaly detection fitting
        if T_corr >= 10:
            n_tf = min(5, T_corr)
            n_af = min(8, n_assets)
            detector = TuckerResidualDetector(n_time_factors=n_tf, n_asset_factors=n_af)
            t_start = time.time()
            try:
                detector.fit(corr_tensor)
                t_detect = time.time() - t_start
            except Exception:
                t_detect = float("nan")
        else:
            t_detect = float("nan")

        results.append({
            "n_assets": n_assets,
            "T": T,
            "T_corr": T_corr,
            "corr_tensor_shape": corr_tensor.shape,
            "mps_fit_time_s": t_mps,
            "mps_compression_ratio": mps_enc.compression_ratio(),
            "anomaly_fit_time_s": t_detect,
        })

    return {"results": results}


# ============================================================================
# Financial compression benchmark (full pipeline)
# ============================================================================

def financial_compression_benchmark(
    n_assets: int = 100,
    T: int = 1000,
    bond_dims: List[int] = [4, 8, 16, 32],
    tucker_ranks: List[int] = [3, 5, 8, 10],
    key: jax.random.KeyArray = jax.random.PRNGKey(99),
) -> Dict[str, Any]:
    """
    Full financial compression benchmark comparing MPS, Tucker, and CP methods.

    Parameters
    ----------
    n_assets : number of assets
    T : time steps
    bond_dims : MPS bond dimensions to test
    tucker_ranks : Tucker ranks to test
    key : random key

    Returns
    -------
    Comprehensive benchmark results
    """
    from .financial_tensors import (
        CorrelationMPS, CorrelationTucker, build_correlation_tensor
    )
    from .tt_decomp import tt_svd, tt_to_dense

    key, subkey = jax.random.split(key)
    _, corr_tensor = synthetic_correlation_tensor(n_assets, T, n_factors=10, key=subkey)
    T_corr, n, _ = corr_tensor.shape

    print(f"Correlation tensor shape: {corr_tensor.shape}")
    print(f"Dense parameters: {corr_tensor.size:,}")

    mps_results = []
    for bond in bond_dims:
        mps_enc = CorrelationMPS(max_bond=bond)
        t_start = time.time()
        mps_enc.fit(corr_tensor[0])
        t_fit = time.time() - t_start

        recon = mps_enc.transform()
        err = float(jnp.linalg.norm(corr_tensor[0] - recon) / (jnp.linalg.norm(corr_tensor[0]) + 1e-10))
        mps_results.append({
            "bond_dim": bond,
            "error": err,
            "compression": mps_enc.compression_ratio(),
            "fit_time_s": t_fit,
        })

    tucker_results = []
    for rank in tucker_ranks:
        t_rank = min(rank, T_corr)
        a_rank = min(rank, n)
        tucker = CorrelationTucker(n_time_factors=t_rank, n_asset_factors=a_rank)
        t_start = time.time()
        tucker.fit(corr_tensor)
        t_fit = time.time() - t_start
        err = tucker.reconstruction_error(corr_tensor)

        core_sz = tucker.core_.size if tucker.core_ is not None else 0
        factor_sz = sum(f.size for f in tucker.factors_) if tucker.factors_ else 0
        comp = corr_tensor.size / (core_sz + factor_sz + 1e-10)

        tucker_results.append({
            "rank": rank,
            "error": err,
            "compression": comp,
            "fit_time_s": t_fit,
        })

    return {
        "n_assets": n_assets,
        "T": T,
        "T_corr": T_corr,
        "dense_params": corr_tensor.size,
        "mps_results": mps_results,
        "tucker_results": tucker_results,
    }


# ============================================================================
# Anomaly detection benchmark
# ============================================================================

def anomaly_detection_benchmark(
    n_assets: int = 50,
    T_normal: int = 300,
    T_anomaly: int = 50,
    n_runs: int = 5,
    key: jax.random.KeyArray = jax.random.PRNGKey(77),
) -> Dict[str, Any]:
    """
    Benchmark anomaly detection precision and recall.

    Generates normal and anomalous correlation tensors and measures
    detection performance.

    Parameters
    ----------
    n_assets : number of assets
    T_normal : normal-period length
    T_anomaly : anomalous-period length
    n_runs : number of random trials
    key : random key

    Returns
    -------
    Detection performance metrics
    """
    from .anomaly_detection import TuckerResidualDetector
    from .financial_tensors import build_correlation_tensor

    precisions = []
    recalls = []
    f1s = []

    for run in range(n_runs):
        key, sk1, sk2, sk3, sk4 = jax.random.split(key, 5)

        # Normal period
        F_n = jax.random.normal(sk1, (T_normal, 3)) * 0.01
        L_n = jax.random.normal(sk2, (n_assets, 3)) * 0.5
        R_n = F_n @ L_n.T + jax.random.normal(sk3, (T_normal, n_assets)) * 0.005
        corr_normal = build_correlation_tensor(R_n, window=40, stride=10)

        # Anomalous period: inject correlation spike
        F_a = jax.random.normal(sk4, (T_anomaly, 3)) * 0.05  # Higher volatility
        L_a = jax.random.normal(sk1, (n_assets, 3)) * 2.0   # Different structure
        R_a = F_a @ L_a.T + jax.random.normal(sk2, (T_anomaly, n_assets)) * 0.05
        corr_anomaly = build_correlation_tensor(R_a, window=40, stride=10)

        if corr_normal.shape[0] < 5 or corr_anomaly.shape[0] < 2:
            continue

        # Fit on normal
        n_tf = min(5, corr_normal.shape[0])
        n_af = min(8, n_assets)
        detector = TuckerResidualDetector(n_time_factors=n_tf, n_asset_factors=n_af)
        try:
            detector.fit(corr_normal)
        except Exception:
            continue

        # Score anomaly period
        scores = detector.predict_scores(corr_anomaly)
        labels = scores > detector.threshold_

        T_a_eval = corr_anomaly.shape[0]
        true_labels = jnp.ones(T_a_eval, dtype=bool)

        tp = float(jnp.sum(labels & true_labels))
        fp = float(jnp.sum(labels & ~true_labels))
        fn = float(jnp.sum(~labels & true_labels))

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "n_runs": n_runs,
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "std_f1": float(np.std(f1s)) if f1s else 0.0,
        "all_precisions": precisions,
        "all_recalls": recalls,
        "all_f1s": f1s,
    }


# ============================================================================
# Riemannian optimizer comparison
# ============================================================================

def optimizer_comparison_benchmark(
    tensor: jnp.ndarray,
    target_rank: int = 8,
    n_steps: int = 50,
    key: jax.random.KeyArray = jax.random.PRNGKey(33),
) -> Dict[str, Any]:
    """
    Compare convergence of different Riemannian optimizers.

    Parameters
    ----------
    tensor : target tensor to approximate
    target_rank : TT-rank budget
    n_steps : optimization steps
    key : random key

    Returns
    -------
    Convergence histories for each optimizer
    """
    from .tt_decomp import tt_svd, tt_to_dense
    from .riemannian_optim import RiemannianGradientDescent, RiemannianAdam, RiemannianConjugateGradient

    # Reference: TT-SVD approximation
    tt_ref = tt_svd(tensor, max_rank=target_rank * 2, cutoff=0.0)
    tt_init = tt_svd(tensor, max_rank=target_rank, cutoff=1e-6)

    def loss_fn(tt):
        recon = tt_to_dense(tt).reshape(tensor.shape)
        return jnp.sum((tensor - recon) ** 2) / (jnp.linalg.norm(tensor) ** 2 + 1e-10)

    results = {}

    # GD
    gd = RiemannianGradientDescent(lr=0.01, max_rank=target_rank)
    _, losses_gd = gd.optimize(tt_init, loss_fn, n_steps=n_steps)
    results["gradient_descent"] = {"losses": losses_gd, "grad_norms": gd.grad_norms_}

    # Adam
    adam = RiemannianAdam(lr=0.001, max_rank=target_rank)
    _, losses_adam = adam.optimize(tt_init, loss_fn, n_steps=n_steps)
    results["riemannian_adam"] = {"losses": losses_adam, "grad_norms": adam.grad_norms_}

    # CG
    cg = RiemannianConjugateGradient(max_rank=target_rank, line_search_lr=0.01)
    _, losses_cg = cg.optimize(tt_init, loss_fn, n_steps=n_steps)
    results["conjugate_gradient"] = {"losses": losses_cg}

    # TT-SVD (reference, single shot)
    t_start = time.time()
    tt_svd_result = tt_svd(tensor, max_rank=target_rank, cutoff=1e-6)
    t_svd = time.time() - t_start
    svd_error = float(loss_fn(tt_svd_result))

    results["tt_svd_reference"] = {
        "final_error": svd_error,
        "time_s": t_svd,
    }

    return results


# ============================================================================
# Timing analysis
# ============================================================================

def timing_analysis(
    n_assets: int = 200,
    T: int = 500,
    n_reps: int = 3,
    key: jax.random.KeyArray = jax.random.PRNGKey(11),
) -> Dict[str, Any]:
    """
    Detailed timing analysis for all major TensorNet operations.

    Parameters
    ----------
    n_assets : number of assets
    T : time steps
    n_reps : number of repetitions per operation
    key : random key

    Returns
    -------
    Per-operation timing statistics
    """
    from .mps import mps_random, mps_inner_product, mps_compress, mps_left_canonicalize
    from .tt_decomp import tt_svd, tt_round, tucker_decomp
    from .financial_tensors import build_correlation_tensor, CorrelationMPS

    key, subkey = jax.random.split(key)
    _, corr_tensor = synthetic_correlation_tensor(n_assets, T, key=subkey)
    T_corr = corr_tensor.shape[0]

    timings = {}

    # MPS inner product
    key, sk1, sk2 = jax.random.split(key, 3)
    n_sites = 8
    mps1 = mps_random(n_sites, 4, 16, sk1)
    mps2 = mps_random(n_sites, 4, 16, sk2)
    times = []
    for _ in range(n_reps):
        t = time.time()
        _ = mps_inner_product(mps1, mps2)
        times.append(time.time() - t)
    timings["mps_inner_product"] = {
        "mean_ms": float(np.mean(times)) * 1000,
        "n_sites": n_sites,
        "bond_dim": 16,
    }

    # TT-SVD on small tensor
    small_tensor = corr_tensor[0]  # (n_assets, n_assets)
    times = []
    for _ in range(n_reps):
        t = time.time()
        _ = tt_svd(small_tensor, max_rank=16)
        times.append(time.time() - t)
    timings["tt_svd_matrix"] = {
        "mean_ms": float(np.mean(times)) * 1000,
        "shape": small_tensor.shape,
    }

    # Tucker decomposition
    if T_corr >= 5:
        small_corr = corr_tensor[:min(30, T_corr)]
        n_use = min(20, n_assets)
        small_corr = small_corr[:, :n_use, :n_use]
        times = []
        for _ in range(n_reps):
            t = time.time()
            _ = tucker_decomp(small_corr, [min(5, small_corr.shape[0]), n_use // 2, n_use // 2], n_iter=5)
            times.append(time.time() - t)
        timings["tucker_decomp"] = {
            "mean_ms": float(np.mean(times)) * 1000,
            "shape": small_corr.shape,
        }

    # MPS compression
    mps3 = mps_random(10, 2, 32, key)
    times = []
    for _ in range(n_reps):
        t = time.time()
        _ = mps_compress(mps3, max_bond=8)
        times.append(time.time() - t)
    timings["mps_compress"] = {
        "mean_ms": float(np.mean(times)) * 1000,
        "original_bond": 32,
        "compressed_bond": 8,
    }

    return {"timings": timings, "n_assets": n_assets, "T": T}


# ============================================================================
# Utility functions
# ============================================================================

def _summarize_results(results: List[Dict]) -> Dict[str, Any]:
    """Compute summary statistics over a list of result dicts."""
    if not results:
        return {}

    # Find common numeric fields
    numeric_fields = [k for k, v in results[0].items() if isinstance(v, (int, float))]
    summary = {}
    for field in numeric_fields:
        vals = [r[field] for r in results if field in r and not math.isnan(r[field])]
        if vals:
            summary[f"{field}_mean"] = float(np.mean(vals))
            summary[f"{field}_min"] = float(np.min(vals))
            summary[f"{field}_max"] = float(np.max(vals))
    return summary


def print_benchmark_report(results: Dict[str, Any], title: str = "Benchmark"):
    """Pretty-print a benchmark report."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    for key, val in results.items():
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
            print(f"\n  {key}:")
            for row in val:
                print("    " + ", ".join(f"{k}: {v:.4g}" if isinstance(v, float) else f"{k}: {v}"
                                        for k, v in row.items()))
        elif isinstance(val, dict):
            print(f"\n  {key}:")
            for k, v in val.items():
                print(f"    {k}: {v:.4g}" if isinstance(v, float) else f"    {k}: {v}")
        else:
            print(f"  {key}: {val}")
    print(f"{'=' * 60}\n")


def run_all_benchmarks(
    quick: bool = True,
    key: jax.random.KeyArray = jax.random.PRNGKey(0),
) -> Dict[str, Any]:
    """
    Run the full TensorNet benchmark suite.

    Parameters
    ----------
    quick : if True, use smaller problem sizes for speed
    key : random key

    Returns
    -------
    All benchmark results
    """
    results = {}

    if quick:
        n_assets = 20
        T = 200
        shapes = [(4, 4, 4)]
        ranks = [2, 4]
        bond_dims = [4, 8]
        tucker_ranks = [3, 5]
    else:
        n_assets = 100
        T = 500
        shapes = [(4, 4, 4), (8, 8, 8), (16, 16, 16)]
        ranks = [2, 4, 8]
        bond_dims = [4, 8, 16, 32]
        tucker_ranks = [3, 5, 8, 10]

    print("Running TT-SVD compression benchmark...")
    key, subkey = jax.random.split(key)
    results["tt_svd"] = benchmark_tt_svd_compression(shapes, ranks, key=subkey)

    print("Running Tucker compression benchmark...")
    corr_shapes = [(T, n_assets, n_assets)]
    key, subkey = jax.random.split(key)
    results["tucker"] = benchmark_tucker_compression(corr_shapes, [3], [5], key=subkey)

    print("Running MPS operations benchmark...")
    key, subkey = jax.random.split(key)
    results["mps_ops"] = benchmark_mps_operations(
        n_sites_list=[4, 8],
        phys_dims=[2, 4],
        bond_dims=[4, 8],
        key=subkey,
    )

    print("Running financial compression benchmark...")
    key, subkey = jax.random.split(key)
    results["financial"] = financial_compression_benchmark(
        n_assets=n_assets, T=T,
        bond_dims=bond_dims[:2],
        tucker_ranks=tucker_ranks[:2],
        key=subkey,
    )

    print("Running anomaly detection benchmark...")
    key, subkey = jax.random.split(key)
    results["anomaly"] = anomaly_detection_benchmark(
        n_assets=20, T_normal=200, T_anomaly=30, n_runs=2, key=subkey
    )

    print("Running timing analysis...")
    key, subkey = jax.random.split(key)
    results["timing"] = timing_analysis(n_assets=min(50, n_assets), T=T, n_reps=2, key=subkey)

    return results
