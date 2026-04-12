"""
experiments.py — Full experiment runners for TensorNet.

Each experiment generates synthetic data, runs compression/analysis,
produces plots, and prints a results table.

Results and plots are saved to aeternus/tensor_net/results/
"""

from __future__ import annotations

import os
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Set results directory
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def generate_correlated_returns(
    n_assets: int,
    n_bars: int,
    n_regimes: int = 3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Generate synthetic multi-asset returns with regime changes.

    Returns
    -------
    returns : array of shape (n_bars, n_assets)
    regime_labels : array of shape (n_bars,) with integer regime IDs
    regime_boundaries : list of bar indices where regimes start
    """
    np.random.seed(seed)
    regime_size = n_bars // n_regimes
    returns_list = []
    regime_labels = []

    regime_configs = []
    # Regime 0: Low vol, mild correlation
    regime_configs.append({
        "vol": 0.01,
        "corr": 0.2,
        "n": regime_size,
    })
    # Regime 1: Medium vol, moderate correlation
    if n_regimes >= 2:
        regime_configs.append({
            "vol": 0.025,
            "corr": 0.5,
            "n": regime_size,
        })
    # Regime 2: High vol, strong correlation (crisis)
    if n_regimes >= 3:
        n_last = n_bars - 2 * regime_size
        regime_configs.append({
            "vol": 0.08,
            "corr": 0.85,
            "n": n_last,
        })

    boundaries = []
    bar = 0
    for r_id, config in enumerate(regime_configs):
        n_r = config["n"]
        vol = config["vol"]
        corr = config["corr"]

        # Build covariance matrix
        cov = vol ** 2 * (corr * np.ones((n_assets, n_assets)) +
                          (1 - corr) * np.eye(n_assets))
        cov += 1e-8 * np.eye(n_assets)  # Regularization

        L = np.linalg.cholesky(cov)
        ret_r = np.random.randn(n_r, n_assets) @ L.T

        # Add fat tails in crisis regime
        if r_id == n_regimes - 1:
            shock_mask = np.random.rand(n_r) < 0.05
            ret_r[shock_mask] *= 4.0

        returns_list.append(ret_r)
        regime_labels.extend([r_id] * n_r)
        boundaries.append(bar)
        bar += n_r

    returns = np.vstack(returns_list)
    return returns, np.array(regime_labels), boundaries


def generate_granger_tensor(
    n_assets: int,
    n_bars: int,
    max_lags: int = 10,
    n_causal_edges: int = 20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate returns with known Granger causal structure.

    Returns
    -------
    returns : array of shape (n_bars, n_assets)
    true_causality : array of shape (n_assets, n_assets, max_lags) — ground truth
    """
    np.random.seed(seed)
    returns = np.zeros((n_bars, n_assets))
    returns[0] = np.random.randn(n_assets) * 0.01

    # Build random sparse causal structure
    causality = np.zeros((n_assets, n_assets, max_lags))
    edges = []
    for _ in range(n_causal_edges):
        i = np.random.randint(n_assets)
        j = np.random.randint(n_assets)
        if i != j:
            lag = np.random.randint(1, max_lags + 1)
            strength = np.random.uniform(0.1, 0.4)
            causality[i, j, lag - 1] = strength
            edges.append((i, j, lag, strength))

    # Generate returns following the causal structure
    for t in range(1, n_bars):
        noise = np.random.randn(n_assets) * 0.01
        causal_effect = np.zeros(n_assets)
        for (i, j, lag, strength) in edges:
            if t >= lag:
                causal_effect[i] += strength * returns[t - lag, j]
        returns[t] = causal_effect + noise

    return returns, causality


# ---------------------------------------------------------------------------
# Experiment 1: Correlation compression
# ---------------------------------------------------------------------------

def experiment_correlation_compression(
    n_assets: int = 30,
    n_bars: int = 1000,
    window: int = 500,
    bond_dims_to_test: List[int] = None,
    save_results: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Generate 1000-bar returns for 30 assets.
    Compress rolling 500-bar correlation matrices as MPS.
    Plot compression error vs bond dimension.
    Show that D=8 captures 95%+ of variance with significant compression.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Correlation Compression")
    print("=" * 70)

    if bond_dims_to_test is None:
        bond_dims_to_test = [1, 2, 4, 8, 16, 32]

    # Generate data
    print(f"Generating {n_bars} bars for {n_assets} assets...")
    returns, regime_labels, boundaries = generate_correlated_returns(
        n_assets, n_bars, seed=seed
    )
    print(f"Regime boundaries: {boundaries}")

    # Compress at different bond dimensions
    from tensor_net.financial_compression import CorrelationMPS
    from tensor_net.mps import mps_compression_analysis

    results_by_bond = {"bond_dims": [], "errors": [], "compression_ratios": [],
                        "variance_explained": []}

    # Use last `window` bars as reference window
    ref_window = returns[-window:]

    print(f"\nCompressing {n_assets}×{n_assets} correlation matrix as MPS:")
    print(f"{'D':>4} | {'Error':>10} | {'Ratio':>10} | {'n_params':>10} | {'Time':>8}")
    print("-" * 50)

    for D in bond_dims_to_test:
        t0 = time.time()
        comp = CorrelationMPS(n_assets, max_bond=D, window=window)
        comp.fit(jnp.array(ref_window))
        t1 = time.time()

        error = comp.compression_error_
        ratio = comp.compression_ratio_
        n_params = comp.mps_.num_params()

        # Variance explained at D components
        var_exp = comp.variance_explained(k=D)

        results_by_bond["bond_dims"].append(D)
        results_by_bond["errors"].append(error)
        results_by_bond["compression_ratios"].append(ratio)
        results_by_bond["variance_explained"].append(var_exp)

        print(f"{D:>4} | {error:>10.6f} | {ratio:>10.2f}x | {n_params:>10,} | {(t1-t0)*1000:>6.1f}ms")

    # Find D where error < 5%
    target_error = 0.05
    good_D = [D for D, err in zip(results_by_bond["bond_dims"], results_by_bond["errors"])
              if err < target_error]
    min_good_D = min(good_D) if good_D else "N/A"
    print(f"\nMinimum D for <5% error: D={min_good_D}")

    # Rolling compression analysis
    print(f"\nRunning rolling compression (D=8, window={window})...")
    comp_rolling = CorrelationMPS(n_assets, max_bond=8, window=window)
    rolling_results = comp_rolling.fit_rolling(jnp.array(returns))

    rolling_errors = [r["error"] for r in rolling_results]
    rolling_timestamps = [r["t"] for r in rolling_results]

    print(f"Rolling compression errors: mean={np.mean(rolling_errors):.4f}, "
          f"max={np.max(rolling_errors):.4f}")

    # Generate plots
    if save_results:
        from tensor_net.visualization import (
            plot_compression_error_vs_ratio,
            plot_bond_dimensions,
            plot_compression_dashboard,
        )

        # Compression tradeoff plot
        plot_compression_error_vs_ratio(
            results_by_bond,
            title=f"Correlation MPS Compression — {n_assets} Assets",
            save_path=str(RESULTS_DIR / "exp1_compression_tradeoff.png"),
            target_error=0.05,
        )
        print(f"Saved: results/exp1_compression_tradeoff.png")

        # Bond dimensions for best compression
        comp_best = CorrelationMPS(n_assets, max_bond=8, window=window)
        comp_best.fit(jnp.array(ref_window))
        plot_bond_dimensions(
            comp_best.mps_,
            title=f"MPS Bond Dimensions (D=8, {n_assets} assets)",
            show_max_bond=8,
            save_path=str(RESULTS_DIR / "exp1_bond_dims.png"),
        )
        print(f"Saved: results/exp1_bond_dims.png")

        # Dashboard
        corr_orig = np.array(comp_best.last_corr_matrix_)
        corr_recon = np.array(comp_best.decompress())
        plot_compression_dashboard(
            corr_orig, corr_recon,
            errors_by_bond=results_by_bond,
            mps=comp_best.mps_,
            title=f"Correlation Compression Dashboard — D=8",
            save_path=str(RESULTS_DIR / "exp1_dashboard.png"),
        )
        print(f"Saved: results/exp1_dashboard.png")

    # Summary table
    print("\n--- RESULTS TABLE ---")
    print(f"{'Bond D':>8} | {'Error':>10} | {'Compression':>12} | {'Var Expl':>10}")
    print("-" * 50)
    for D, err, ratio, ve in zip(
        results_by_bond["bond_dims"],
        results_by_bond["errors"],
        results_by_bond["compression_ratios"],
        results_by_bond["variance_explained"],
    ):
        print(f"{D:>8} | {err:>10.4f} | {ratio:>10.1f}x | {ve*100:>9.1f}%")

    return {
        "returns": returns,
        "regime_labels": regime_labels,
        "regime_boundaries": boundaries,
        "compression_results": results_by_bond,
        "rolling_errors": rolling_errors,
        "rolling_timestamps": rolling_timestamps,
        "min_good_D": min_good_D,
    }


# ---------------------------------------------------------------------------
# Experiment 2: Crisis anomaly detection
# ---------------------------------------------------------------------------

def experiment_crisis_anomaly_detection(
    n_assets: int = 30,
    n_bars: int = 1000,
    crisis_bar: int = 825,
    max_bond: int = 8,
    detection_window: int = 20,
    save_results: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Inject crisis at bar 825 (matching Event Horizon setup).
    Show that MPS reconstruction error spikes 50-100 bars before crisis.
    Compare vs PCA-based anomaly detector.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Crisis Anomaly Detection")
    print("=" * 70)
    print(f"n_assets={n_assets}, n_bars={n_bars}, crisis_bar={crisis_bar}")

    np.random.seed(seed)

    # Generate normal-period returns (first 800 bars)
    n_normal = 800
    n_crisis = n_bars - n_normal

    # Normal regime: low vol, moderate correlation
    cov_normal = 0.01 ** 2 * (0.3 * np.ones((n_assets, n_assets)) +
                                0.7 * np.eye(n_assets))
    cov_normal += 1e-8 * np.eye(n_assets)
    L_normal = np.linalg.cholesky(cov_normal)
    returns_normal = np.random.randn(n_normal, n_assets) @ L_normal.T

    # Crisis regime: high vol, very high correlation, fat tails
    cov_crisis = 0.08 ** 2 * (0.85 * np.ones((n_assets, n_assets)) +
                               0.15 * np.eye(n_assets))
    cov_crisis += 1e-8 * np.eye(n_assets)
    L_crisis = np.linalg.cholesky(cov_crisis)
    returns_crisis = np.random.randn(n_crisis, n_assets) @ L_crisis.T

    # Fat tails
    shock_mask = np.random.rand(n_crisis) < 0.1
    returns_crisis[shock_mask] *= 5.0

    # Gradual regime shift: 50-bar transition starting at crisis_bar - 800 (relative)
    crisis_start_rel = crisis_bar - n_normal  # relative position in crisis period
    transition_window = 50

    # Pre-crisis stress: linearly interpolate between normal and crisis
    for t in range(max(0, crisis_start_rel - transition_window), crisis_start_rel):
        frac = (t - (crisis_start_rel - transition_window)) / transition_window
        # Blend covariance
        L_blend = (1 - frac) * L_normal + frac * L_crisis
        returns_crisis[t] = np.random.randn(n_assets) @ L_blend.T

    # Combine
    returns = np.vstack([returns_normal, returns_crisis])

    print(f"Returns shape: {returns.shape}")
    print(f"Normal period: bars 0-{n_normal}")
    print(f"Crisis period: bars {n_normal}-{n_bars}")
    print(f"Crisis injection at: bar {crisis_bar}")

    # Fit MPS anomaly detector on normal period
    from tensor_net.financial_compression import AnomalyDetector

    print(f"\nFitting MPS anomaly detector on bars 0-{n_normal // 2}...")
    detector = AnomalyDetector(
        n_assets=n_assets,
        max_bond=max_bond,
        window=min(500, n_normal // 2),
        detection_window=detection_window,
        z_score_threshold=2.5,
    )
    detector.fit_baseline(jnp.array(returns[:n_normal // 2]))

    # Score rolling windows
    print("Computing anomaly scores over full period...")
    mps_scores, score_times = detector.score_sequence(
        jnp.array(returns), step=5
    )
    mps_scores_np = np.array(mps_scores)
    score_times_np = np.array(score_times)

    # PCA comparison
    print("Running PCA comparison...")
    pca_comparison = detector.compare_pca_detector(jnp.array(returns), n_components=5)

    # Analyze: when does MPS score first exceed threshold?
    threshold = 2.5
    mps_alert_times = score_times_np[mps_scores_np > threshold]
    pca_alert_times = score_times_np[np.array(pca_comparison["pca_scores"]) > threshold]

    mps_first_alert = int(mps_alert_times.min()) if len(mps_alert_times) > 0 else n_bars
    pca_first_alert = int(pca_alert_times.min()) if len(pca_alert_times) > 0 else n_bars

    lead_mps = crisis_bar - mps_first_alert
    lead_pca = crisis_bar - pca_first_alert

    print(f"\nMPS first alert: bar {mps_first_alert} ({lead_mps} bars before crisis)")
    print(f"PCA first alert: bar {pca_first_alert} ({lead_pca} bars before crisis)")
    print(f"MPS advantage: {lead_mps - lead_pca} bars earlier warning")

    # Score stats in pre-crisis window
    pre_crisis_mask = (score_times_np >= crisis_bar - 100) & (score_times_np < crisis_bar)
    post_crisis_mask = score_times_np >= crisis_bar

    if pre_crisis_mask.any():
        pre_score = float(np.mean(mps_scores_np[pre_crisis_mask]))
        print(f"MPS mean score 100 bars before crisis: {pre_score:.3f}")
    if post_crisis_mask.any():
        post_score = float(np.mean(mps_scores_np[post_crisis_mask]))
        print(f"MPS mean score during crisis: {post_score:.3f}")

    # Generate plots
    if save_results:
        from tensor_net.visualization import plot_anomaly_scores
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        # MPS anomaly scores
        plot_anomaly_scores(
            scores=mps_scores_np,
            returns=returns,
            timestamps=score_times_np,
            crisis_bars=[crisis_bar],
            title="MPS Anomaly Detector — Crisis Detection",
            save_path=str(RESULTS_DIR / "exp2_mps_anomaly.png"),
            threshold=threshold,
        )
        print(f"Saved: results/exp2_mps_anomaly.png")

        # MPS vs PCA comparison
        fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                                  facecolor="#0a0a0f", sharex=True)
        for ax in axes:
            ax.set_facecolor("#0d1117")

        axes[0].plot(score_times_np, mps_scores_np, color="#58a6ff", linewidth=1.2)
        axes[0].axhline(threshold, color="#f85149", linestyle="--", linewidth=1.2)
        axes[0].axvline(crisis_bar, color="#f85149", linewidth=2, label="Crisis start")
        axes[0].set_ylabel("MPS z-score", color="#c9d1d9")
        axes[0].set_title("MPS vs PCA Anomaly Detection", color="#f0f6fc", fontsize=12)
        axes[0].legend(fontsize=9)
        axes[0].tick_params(colors="#8b949e")
        axes[0].grid(alpha=0.2)

        pca_scores = np.array(pca_comparison["pca_scores"])
        axes[1].plot(score_times_np[:len(pca_scores)], pca_scores,
                     color="#3fb950", linewidth=1.2)
        axes[1].axhline(threshold, color="#f85149", linestyle="--", linewidth=1.2)
        axes[1].axvline(crisis_bar, color="#f85149", linewidth=2, label="Crisis start")
        axes[1].set_ylabel("PCA z-score", color="#c9d1d9")
        axes[1].set_xlabel("Bar", color="#c9d1d9")
        axes[1].legend(fontsize=9)
        axes[1].tick_params(colors="#8b949e")
        axes[1].grid(alpha=0.2)

        fig.tight_layout()
        fig.savefig(str(RESULTS_DIR / "exp2_mps_vs_pca.png"), dpi=150,
                    bbox_inches="tight", facecolor="#0a0a0f")
        plt.close(fig)
        print(f"Saved: results/exp2_mps_vs_pca.png")

    # Results table
    print("\n--- RESULTS TABLE ---")
    print(f"{'Metric':<40} | {'MPS':>10} | {'PCA':>10}")
    print("-" * 65)
    print(f"{'First alert bar':<40} | {mps_first_alert:>10} | {pca_first_alert:>10}")
    print(f"{'Lead time (bars before crisis)':<40} | {lead_mps:>10} | {lead_pca:>10}")
    print(f"{'Advantage':<40} | {lead_mps - lead_pca:>10} | {'--':>10}")

    return {
        "returns": returns,
        "mps_scores": mps_scores_np,
        "pca_scores": pca_comparison["pca_scores"],
        "score_times": score_times_np,
        "crisis_bar": crisis_bar,
        "mps_first_alert": mps_first_alert,
        "pca_first_alert": pca_first_alert,
        "lead_mps": lead_mps,
        "lead_pca": lead_pca,
    }


# ---------------------------------------------------------------------------
# Experiment 3: Quantum kernel regime classification
# ---------------------------------------------------------------------------

def experiment_quantum_kernel_regime(
    n_assets: int = 15,
    n_bars: int = 800,
    n_regimes: int = 4,
    max_bond_kernel: int = 8,
    n_qubits: int = 6,
    save_results: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Classify 8 market regimes using quantum kernel SVM.
    Compare vs RBF kernel SVM on raw features.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Quantum Kernel Regime Classification")
    print("=" * 70)

    np.random.seed(seed)

    # Generate data with multiple regimes
    n_per_regime = n_bars // n_regimes
    X_all = []
    y_all = []

    regime_params = []
    for r in range(n_regimes):
        corr_val = 0.1 + 0.2 * r
        vol_val = 0.01 * (1 + r)
        regime_params.append({"corr": corr_val, "vol": vol_val})

    for r in range(n_regimes):
        params = regime_params[r]
        cov = params["vol"] ** 2 * (
            params["corr"] * np.ones((n_assets, n_assets)) +
            (1 - params["corr"]) * np.eye(n_assets)
        ) + 1e-8 * np.eye(n_assets)
        L = np.linalg.cholesky(cov)
        ret = np.random.randn(n_per_regime, n_assets) @ L.T
        X_all.append(ret)
        y_all.extend([r] * n_per_regime)

    X = np.vstack(X_all)
    y = np.array(y_all)

    # Build features: rolling statistics (mean, std, skewness, kurtosis)
    # Use 20-bar windows
    window_feat = 20
    X_features = []
    y_features = []

    for t in range(window_feat, len(X)):
        window = X[t - window_feat:t]
        # Statistical features
        feat = np.concatenate([
            window.mean(axis=0),
            window.std(axis=0),
            np.abs(window).mean(axis=0),
        ])
        X_features.append(feat)
        y_features.append(y[t])

    X_features = np.array(X_features, dtype=np.float32)
    y_features = np.array(y_features)

    # Normalize features
    X_mean = X_features.mean(axis=0)
    X_std = X_features.std(axis=0) + 1e-8
    X_features_norm = (X_features - X_mean) / X_std

    # Train/test split (stratified)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    for train_idx, test_idx in sss.split(X_features_norm, y_features):
        X_train, X_test = X_features_norm[train_idx], X_features_norm[test_idx]
        y_train, y_test = y_features[train_idx], y_features[test_idx]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}, n_regimes: {n_regimes}")
    print(f"Feature dim: {X_train.shape[1]}")

    # Subsample for quantum kernel (computationally expensive)
    max_qk_samples = 200
    if len(X_train) > max_qk_samples:
        idx_sub = np.random.choice(len(X_train), max_qk_samples, replace=False)
        X_train_qk = X_train[idx_sub, :n_qubits * 2]  # Use first 2*n_qubits features
        y_train_qk = y_train[idx_sub]
    else:
        X_train_qk = X_train[:, :n_qubits * 2]
        y_train_qk = y_train

    max_test_samples = 100
    if len(X_test) > max_test_samples:
        idx_test = np.random.choice(len(X_test), max_test_samples, replace=False)
        X_test_qk = X_test[idx_test, :n_qubits * 2]
        y_test_qk = y_test[idx_test]
    else:
        X_test_qk = X_test[:, :n_qubits * 2]
        y_test_qk = y_test

    # Quantum kernel SVM
    print(f"\nTraining Quantum Kernel SVM (n_qubits={n_qubits}, "
          f"n_train={len(X_train_qk)})...")
    t0 = time.time()

    from tensor_net.quantum_inspired import QuantumKernel
    qk = QuantumKernel(n_qubits=n_qubits, max_bond=max_bond_kernel,
                       encoding="angle", n_encoding_layers=2)

    try:
        qk_svm = qk.fit_svm(
            jnp.array(X_train_qk),
            y_train_qk,
        )
        qk_acc = qk_svm.accuracy(jnp.array(X_test_qk), y_test_qk)
        qk_time = time.time() - t0
        print(f"Quantum kernel SVM accuracy: {qk_acc:.4f} (time: {qk_time:.1f}s)")
    except Exception as e:
        print(f"Quantum kernel failed: {e}")
        qk_acc = 0.0
        qk_time = 0.0

    # RBF kernel SVM (baseline)
    print("\nTraining RBF kernel SVM (baseline)...")
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    rbf_svm = SVC(kernel="rbf", C=1.0, gamma="scale")
    rbf_svm.fit(X_train, y_train)
    rbf_acc = float(rbf_svm.score(X_test, y_test))
    rbf_time = time.time() - t0
    print(f"RBF SVM accuracy: {rbf_acc:.4f} (time: {rbf_time:.1f}s)")

    # Random forest baseline
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf.fit(X_train, y_train)
    rf_acc = float(rf.score(X_test, y_test))
    print(f"Random Forest accuracy: {rf_acc:.4f}")

    # Quantum kernel matrix visualization
    if save_results and len(X_train_qk) <= 100:
        from tensor_net.visualization import plot_quantum_kernel_matrix
        K_train = np.array(qk.kernel_matrix(jnp.array(X_train_qk)))
        regime_names = [f"Regime {r}" for r in range(n_regimes)]
        plot_quantum_kernel_matrix(
            K_train,
            labels=y_train_qk,
            label_names=regime_names,
            title=f"Quantum Kernel Matrix — {n_regimes} Market Regimes",
            save_path=str(RESULTS_DIR / "exp3_quantum_kernel_matrix.png"),
        )
        print(f"Saved: results/exp3_quantum_kernel_matrix.png")

    # Results table
    print("\n--- RESULTS TABLE ---")
    print(f"{'Method':<25} | {'Accuracy':>10} | {'Time (s)':>10}")
    print("-" * 50)
    print(f"{'Quantum Kernel SVM':<25} | {qk_acc:>10.4f} | {qk_time:>10.1f}")
    print(f"{'RBF Kernel SVM':<25} | {rbf_acc:>10.4f} | {rbf_time:>10.1f}")
    print(f"{'Random Forest':<25} | {rf_acc:>10.4f} | {'N/A':>10}")

    return {
        "qk_accuracy": qk_acc,
        "rbf_accuracy": rbf_acc,
        "rf_accuracy": rf_acc,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# Experiment 4: Causal tensor compression
# ---------------------------------------------------------------------------

def experiment_causal_tensor_compression(
    n_assets: int = 10,
    n_bars: int = 500,
    max_lags: int = 8,
    bond_dims: List[int] = None,
    save_results: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Compress N×N×lag Granger tensor, show structure preserved at D=4.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Causal Tensor Compression")
    print("=" * 70)
    print(f"n_assets={n_assets}, max_lags={max_lags}")

    if bond_dims is None:
        bond_dims = [1, 2, 4, 8]

    # Generate returns with causal structure
    returns, true_causality = generate_granger_tensor(
        n_assets, n_bars, max_lags, n_causal_edges=15, seed=seed
    )
    print(f"True causal tensor shape: {true_causality.shape}")
    print(f"Non-zero entries: {np.count_nonzero(true_causality)}")

    # Compress using CausalityTensor at different bond dims
    from tensor_net.financial_compression import CausalityTensor
    from tensor_net.tensor_train import tt_to_dense

    results = {"bond_dims": [], "errors": [], "compression_ratios": [],
                "structure_preserved": []}

    print(f"\n{'D':>4} | {'TT Error':>10} | {'Ratio':>10} | {'Structure %':>12}")
    print("-" * 50)

    for D in bond_dims:
        ct = CausalityTensor(n_assets, max_lags, max_bond=D)
        ct.fit(jnp.array(returns))

        error = ct.compression_error_
        ratio = ct.compression_ratio_

        # Check structure preservation: are the same dominant edges preserved?
        reconstructed_tensor = np.array(tt_to_dense(ct.tt_))
        true_flat = true_causality.reshape(-1)
        recon_flat = reconstructed_tensor.reshape(-1)

        # Top-10 edges by magnitude: overlap
        true_top = set(np.argsort(np.abs(true_flat))[-10:])
        recon_top = set(np.argsort(np.abs(recon_flat))[-10:])
        structure_preserved = len(true_top & recon_top) / 10.0

        results["bond_dims"].append(D)
        results["errors"].append(error)
        results["compression_ratios"].append(ratio)
        results["structure_preserved"].append(structure_preserved)

        print(f"{D:>4} | {error:>10.6f} | {ratio:>10.1f}x | {structure_preserved*100:>11.1f}%")

    # Plot causality tensor heatmap
    if save_results:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        ct_best = CausalityTensor(n_assets, max_lags, max_bond=4)
        ct_best.fit(jnp.array(returns))
        recon = np.array(tt_to_dense(ct_best.tt_))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                                  facecolor="#0a0a0f")

        for ax_i, (data, title) in enumerate([
            (np.sum(np.abs(true_causality), axis=2), "True Causality (lag-sum)"),
            (np.sum(np.abs(recon), axis=2), "Reconstructed (D=4, lag-sum)"),
            (np.abs(np.sum(np.abs(true_causality), axis=2) -
                    np.sum(np.abs(recon), axis=2)), "Absolute Difference"),
        ]):
            axes[ax_i].set_facecolor("#0d1117")
            im = axes[ax_i].imshow(data, cmap="hot", aspect="auto")
            plt.colorbar(im, ax=axes[ax_i])
            axes[ax_i].set_title(title, color="#f0f6fc", fontsize=10)
            axes[ax_i].set_xlabel("Asset j (cause)", color="#c9d1d9")
            axes[ax_i].set_ylabel("Asset i (effect)", color="#c9d1d9")
            axes[ax_i].tick_params(colors="#8b949e")

        fig.suptitle(f"Granger Causality Tensor Compression — {n_assets} Assets, {max_lags} Lags",
                     color="#f0f6fc", fontsize=12)
        fig.tight_layout()
        fig.savefig(str(RESULTS_DIR / "exp4_causality_tensor.png"), dpi=150,
                    bbox_inches="tight", facecolor="#0a0a0f")
        plt.close(fig)
        print(f"\nSaved: results/exp4_causality_tensor.png")

    print("\n--- RESULTS TABLE ---")
    print(f"{'D':>4} | {'Error':>10} | {'Compression':>12} | {'Structure %':>12}")
    print("-" * 50)
    for D, err, ratio, sp in zip(
        results["bond_dims"], results["errors"],
        results["compression_ratios"], results["structure_preserved"]
    ):
        print(f"{D:>4} | {err:>10.4f} | {ratio:>10.1f}x | {sp*100:>11.1f}%")

    return {
        "returns": returns,
        "true_causality": true_causality,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Experiment 5: Portfolio VQE
# ---------------------------------------------------------------------------

def experiment_portfolio_vqe(
    n_assets: int = 8,
    n_bars: int = 500,
    n_layers: int = 3,
    n_steps: int = 150,
    save_results: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Use variational MPS circuit to find minimum variance portfolio.
    Compare vs Markowitz on synthetic data.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Variational Portfolio Optimization (VQE-style)")
    print("=" * 70)
    print(f"n_assets={n_assets}, n_layers={n_layers}, n_steps={n_steps}")

    np.random.seed(seed)

    # Generate correlated returns
    cov_true = 0.02 ** 2 * (0.4 * np.ones((n_assets, n_assets)) +
                              0.6 * np.eye(n_assets))
    cov_true += 1e-8 * np.eye(n_assets)
    L = np.linalg.cholesky(cov_true)
    returns = np.random.randn(n_bars, n_assets) @ L.T
    expected_rets = returns.mean(axis=0)

    # Estimate empirical covariance
    cov_emp = np.cov(returns.T)
    print(f"Empirical covariance estimated from {n_bars} observations")

    # Markowitz minimum variance portfolio
    from scipy.optimize import minimize
    def mv_obj(w):
        return w @ cov_emp @ w

    def mv_constraint(w):
        return w.sum() - 1.0

    w0 = np.ones(n_assets) / n_assets
    result_mw = minimize(
        mv_obj, w0,
        method="SLSQP",
        constraints={"type": "eq", "fun": mv_constraint},
        bounds=[(0, 1)] * n_assets,
    )
    w_markowitz = result_mw.x
    var_markowitz = float(w_markowitz @ cov_emp @ w_markowitz)
    ret_markowitz = float(w_markowitz @ expected_rets)

    print(f"\nMarkowitz minimum variance:")
    print(f"  Variance: {var_markowitz:.6f}  Return: {ret_markowitz:.6f}")

    # Equal weight baseline
    w_equal = np.ones(n_assets) / n_assets
    var_equal = float(w_equal @ cov_emp @ w_equal)
    ret_equal = float(w_equal @ expected_rets)
    print(f"\nEqual weight baseline:")
    print(f"  Variance: {var_equal:.6f}  Return: {ret_equal:.6f}")

    # VQE-style MPS portfolio optimizer
    print(f"\nRunning VQE portfolio optimizer (n_layers={n_layers})...")
    from tensor_net.quantum_inspired import VariationalPortfolioOptimizer

    t0 = time.time()
    vqe = VariationalPortfolioOptimizer(
        n_assets=n_assets,
        n_layers=n_layers,
        max_bond=8,
        n_steps=n_steps,
        lr=0.02,
    )
    vqe_result = vqe.optimize(
        jnp.array(cov_emp.astype(np.float32)),
        expected_returns=jnp.array(expected_rets.astype(np.float32)),
        risk_aversion=1.0,
    )
    vqe_time = time.time() - t0

    w_vqe = np.array(vqe_result["weights"])
    var_vqe = float(w_vqe @ cov_emp @ w_vqe)
    ret_vqe = float(w_vqe @ expected_rets)

    print(f"VQE result (time: {vqe_time:.1f}s):")
    print(f"  Variance: {var_vqe:.6f}  Return: {ret_vqe:.6f}")
    print(f"  Steps taken: {vqe_result['n_steps_taken']}")

    # Compare
    var_improvement_vs_equal = (var_equal - var_vqe) / var_equal * 100
    var_gap_vs_markowitz = (var_vqe - var_markowitz) / var_markowitz * 100

    print(f"\nVQE vs Equal weight: {var_improvement_vs_equal:.1f}% variance reduction")
    print(f"VQE vs Markowitz gap: {var_gap_vs_markowitz:.1f}%")

    # Plot results
    if save_results:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                                  facecolor="#0a0a0f")

        # Portfolio weights comparison
        ax = axes[0]
        ax.set_facecolor("#0d1117")
        x = np.arange(n_assets)
        width = 0.25
        ax.bar(x - width, w_markowitz, width, label="Markowitz", color="#58a6ff", alpha=0.85)
        ax.bar(x, w_vqe, width, label="VQE", color="#3fb950", alpha=0.85)
        ax.bar(x + width, w_equal, width, label="Equal", color="#ffa657", alpha=0.85)
        ax.set_xlabel("Asset", color="#c9d1d9")
        ax.set_ylabel("Weight", color="#c9d1d9")
        ax.set_title("Portfolio Weights", color="#f0f6fc")
        ax.legend(fontsize=8)
        ax.tick_params(colors="#8b949e")
        ax.grid(axis="y", alpha=0.3)

        # VQE loss curve
        ax = axes[1]
        ax.set_facecolor("#0d1117")
        losses = vqe_result["losses"]
        ax.semilogy(losses, color="#d2a8ff", linewidth=1.5)
        ax.set_xlabel("Optimization step", color="#c9d1d9")
        ax.set_ylabel("Portfolio variance", color="#c9d1d9")
        ax.set_title("VQE Convergence", color="#f0f6fc")
        ax.tick_params(colors="#8b949e")
        ax.grid(alpha=0.3)

        # Risk-return scatter
        ax = axes[2]
        ax.set_facecolor("#0d1117")
        methods = ["Equal", "Markowitz", "VQE"]
        variances = [var_equal, var_markowitz, var_vqe]
        returns_vals = [ret_equal, ret_markowitz, ret_vqe]
        colors_m = ["#ffa657", "#58a6ff", "#3fb950"]

        for m, v, r, c in zip(methods, variances, returns_vals, colors_m):
            ax.scatter([v], [r], s=150, color=c, label=m, zorder=5, edgecolors="white", linewidth=0.5)
            ax.annotate(m, (v, r), textcoords="offset points",
                       xytext=(5, 5), fontsize=9, color=c)

        ax.set_xlabel("Portfolio Variance", color="#c9d1d9")
        ax.set_ylabel("Expected Return", color="#c9d1d9")
        ax.set_title("Risk-Return Comparison", color="#f0f6fc")
        ax.legend(fontsize=8)
        ax.tick_params(colors="#8b949e")
        ax.grid(alpha=0.3)

        fig.suptitle(f"VQE Portfolio Optimization — {n_assets} Assets",
                     color="#f0f6fc", fontsize=12)
        fig.tight_layout()
        fig.savefig(str(RESULTS_DIR / "exp5_portfolio_vqe.png"), dpi=150,
                    bbox_inches="tight", facecolor="#0a0a0f")
        plt.close(fig)
        print(f"\nSaved: results/exp5_portfolio_vqe.png")

    # Results table
    print("\n--- RESULTS TABLE ---")
    print(f"{'Method':<15} | {'Variance':>12} | {'Return':>12} | {'Sharpe (proxy)':>15}")
    print("-" * 65)
    for method, var, ret in [
        ("Equal", var_equal, ret_equal),
        ("Markowitz", var_markowitz, ret_markowitz),
        ("VQE", var_vqe, ret_vqe),
    ]:
        sharpe = ret / (math.sqrt(var) + 1e-12)
        print(f"{method:<15} | {var:>12.6f} | {ret:>12.6f} | {sharpe:>15.4f}")

    return {
        "returns": returns,
        "cov_emp": cov_emp,
        "weights_markowitz": w_markowitz,
        "weights_vqe": w_vqe,
        "weights_equal": w_equal,
        "variance_markowitz": var_markowitz,
        "variance_vqe": var_vqe,
        "variance_equal": var_equal,
        "vqe_losses": vqe_result["losses"],
        "var_improvement_vs_equal": var_improvement_vs_equal,
        "var_gap_vs_markowitz": var_gap_vs_markowitz,
    }


# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------

def run_all_experiments(save_results: bool = True) -> Dict:
    """
    Run all TensorNet experiments and save results.

    Returns dict with all experiment results.
    """
    print("\n" + "#" * 70)
    print("# TENSORNET EXPERIMENTS — PROJECT AETERNUS MODULE 3")
    print("#" * 70)
    print(f"Results directory: {RESULTS_DIR}")

    all_results = {}

    try:
        print("\n[1/5] Running correlation compression experiment...")
        all_results["exp1"] = experiment_correlation_compression(
            n_assets=20, n_bars=800, window=400, save_results=save_results
        )
    except Exception as e:
        print(f"Exp 1 failed: {e}")
        all_results["exp1"] = {"error": str(e)}

    try:
        print("\n[2/5] Running crisis anomaly detection experiment...")
        all_results["exp2"] = experiment_crisis_anomaly_detection(
            n_assets=20, n_bars=1000, crisis_bar=825, save_results=save_results
        )
    except Exception as e:
        print(f"Exp 2 failed: {e}")
        all_results["exp2"] = {"error": str(e)}

    try:
        print("\n[3/5] Running quantum kernel regime classification experiment...")
        all_results["exp3"] = experiment_quantum_kernel_regime(
            n_assets=10, n_bars=600, n_regimes=4, save_results=save_results
        )
    except Exception as e:
        print(f"Exp 3 failed: {e}")
        all_results["exp3"] = {"error": str(e)}

    try:
        print("\n[4/5] Running causal tensor compression experiment...")
        all_results["exp4"] = experiment_causal_tensor_compression(
            n_assets=8, n_bars=400, max_lags=6, save_results=save_results
        )
    except Exception as e:
        print(f"Exp 4 failed: {e}")
        all_results["exp4"] = {"error": str(e)}

    try:
        print("\n[5/5] Running VQE portfolio optimization experiment...")
        all_results["exp5"] = experiment_portfolio_vqe(
            n_assets=6, n_bars=300, n_layers=2, n_steps=100, save_results=save_results
        )
    except Exception as e:
        print(f"Exp 5 failed: {e}")
        all_results["exp5"] = {"error": str(e)}

    print("\n" + "#" * 70)
    print("# ALL EXPERIMENTS COMPLETE")
    print(f"# Results saved to: {RESULTS_DIR}")
    print("#" * 70)

    return all_results


if __name__ == "__main__":
    run_all_experiments()
