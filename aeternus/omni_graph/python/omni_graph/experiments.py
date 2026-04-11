"""
experiments.py — Benchmark experiments for the Omni-Graph module.

Experiments:
    run_wormhole_detection_experiment     Synthetic crisis wormhole detection.
    run_regime_classification_experiment  Market regime classification from topology.
    run_crisis_prediction_experiment      Crisis timing from Ricci curvature trajectory.
    run_link_prediction_benchmark         Link prediction vs. correlation baseline.

All experiments save results to CSV and plots to PNG in the output directory.
"""

from __future__ import annotations

import os
import math
import time
import datetime
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from omni_graph.financial_graphs import (
    build_correlation_graph,
    GraphEvolution,
)
from omni_graph.dynamic_gnn import (
    EvolutionaryGNN,
    make_synthetic_snapshot,
)
from omni_graph.edge_prediction import (
    WormholeDetector,
    GraphDiffusion,
    RicciFlowGNN,
    evaluate_link_prediction,
)
from omni_graph.regime_gnn import (
    GraphRegimeDetector,
    WassersteinGraphKernel,
    CrisisEarlyWarning,
    RegimeTransitionPredictor,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ── Synthetic data generation ─────────────────────────────────────────────────

def generate_synthetic_returns(
    n_assets: int = 20,
    n_periods: int = 500,
    n_regimes: int = 4,
    crisis_length: int = 50,
    crisis_at: int = 350,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic return time series with known regimes and a crisis.

    Returns:
        returns:   (T, N) return matrix.
        regimes:   (T,) regime labels (0=bull, 1=bear, 2=neutral, 3=crisis).
        crisis_mask: (T,) boolean mask for crisis periods.
    """
    rng = np.random.RandomState(seed)

    returns = np.zeros((n_periods, n_assets))
    regimes = np.zeros(n_periods, dtype=int)

    # Define regime parameters
    regime_params = {
        0: {"mu": 0.001, "sigma": 0.01, "corr": 0.2},   # bull: low vol, low corr
        1: {"mu": -0.001, "sigma": 0.02, "corr": 0.3},  # bear: med vol, med corr
        2: {"mu": 0.0, "sigma": 0.015, "corr": 0.15},   # neutral: low corr
        3: {"mu": -0.003, "sigma": 0.04, "corr": 0.8},  # crisis: high vol, high corr
    }

    # Generate regime sequence (Markov chain)
    transition = np.array([
        [0.95, 0.03, 0.02, 0.00],  # from bull
        [0.05, 0.88, 0.05, 0.02],  # from bear
        [0.10, 0.05, 0.83, 0.02],  # from neutral
        [0.02, 0.10, 0.08, 0.80],  # from crisis (sticky)
    ])

    current_regime = 0
    for t in range(n_periods):
        # Force crisis at specified period
        if crisis_at <= t < crisis_at + crisis_length:
            current_regime = 3
        elif t == crisis_at + crisis_length:
            current_regime = 1  # post-crisis bear
        else:
            probs = transition[current_regime]
            # Don't transition to crisis unless forced
            probs_adj = probs.copy()
            probs_adj[3] = 0.0
            probs_adj = probs_adj / probs_adj.sum()
            current_regime = rng.choice(n_regimes, p=probs_adj)

        regimes[t] = current_regime
        p = regime_params[current_regime]

        # Generate correlated returns
        corr_matrix = p["corr"] * np.ones((n_assets, n_assets)) + (1 - p["corr"]) * np.eye(n_assets)
        L = np.linalg.cholesky(corr_matrix + 1e-8 * np.eye(n_assets))
        z = rng.randn(n_assets)
        returns[t] = p["mu"] + p["sigma"] * (L @ z)

    # Add wormhole effect: before crisis, introduce sudden cross-cluster edges
    # by creating sudden correlations between previously uncorrelated groups
    pre_crisis_start = crisis_at - 30
    pre_crisis_end = crisis_at
    for t in range(pre_crisis_start, pre_crisis_end):
        # Amplify correlation between first and second half of assets
        extra_corr = 0.5 * (t - pre_crisis_start) / 30.0  # ramps up
        half = n_assets // 2
        noise = rng.randn(n_assets) * 0.02
        returns[t, :half] += extra_corr * returns[t, half:].mean()
        returns[t] += noise

    crisis_mask = regimes == 3
    return returns, regimes, crisis_mask


def build_synthetic_graphs(
    returns: np.ndarray,
    window: int = 40,
    step: int = 5,
    threshold: float = 0.3,
) -> Tuple[List[Data], List[int], List[int]]:
    """Build graph snapshots from synthetic returns."""
    T, N = returns.shape
    snapshots = []
    timestamps = []

    for start in range(0, T - window, step):
        end = start + window
        snap = build_correlation_graph(returns[start:end], threshold)
        snapshots.append(snap)
        timestamps.append(end)

    return snapshots, timestamps, list(range(len(snapshots)))


def approximate_ricci_numpy(snap: Data) -> np.ndarray:
    """Approximate Ollivier-Ricci curvature in pure Python (for experiments).

    Uses Forman-Ricci as a fast proxy (no Rust dependency required in tests).
    kappa_Forman(u,v) = w(u,v) * [2 - d(u) - d(v) + 3 * w(u,v) / (w_max_u + w_max_v)]
    """
    n = snap.x.size(0)
    ei = snap.edge_index.numpy()
    n_edges = ei.shape[1]

    if n_edges == 0:
        return np.zeros(0)

    ew = snap.edge_attr[:, 0].numpy() if snap.edge_attr is not None else np.ones(n_edges)

    # Degree (sum of weights) per node
    degree = np.zeros(n)
    max_weight = np.zeros(n)
    for e in range(n_edges):
        u, v = ei[0, e], ei[1, e]
        degree[u] += ew[e]
        degree[v] += ew[e]
        max_weight[u] = max(max_weight[u], ew[e])
        max_weight[v] = max(max_weight[v], ew[e])

    curvatures = np.zeros(n_edges)
    for e in range(n_edges):
        u, v = ei[0, e], ei[1, e]
        w = ew[e]
        denom = max_weight[u] + max_weight[v]
        if denom < 1e-10:
            denom = 1.0
        curv = w * (2.0 - degree[u] - degree[v] + 3.0 * w / denom)
        curvatures[e] = curv

    return curvatures


# ── Experiment 1: Wormhole detection ─────────────────────────────────────────

def run_wormhole_detection_experiment(
    output_dir: str = "outputs",
    n_assets: int = 20,
    n_periods: int = 500,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Detect wormhole edges before and during a synthetic crisis.

    Generates synthetic returns with a known crisis at period 350.
    Trains WormholeDetector on pre-crisis data, then scores all subsequent
    snapshots.

    Expected outcome:
    - Anomaly scores should spike in the 30 periods before the crisis onset.
    - True positives: wormhole edges connecting previously disconnected clusters.

    Returns:
        Results dict with precision, recall, AUC, and per-step scores.
    """
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    if verbose:
        print("\n=== Experiment 1: Wormhole Detection ===")

    # Generate data
    returns, regimes, crisis_mask = generate_synthetic_returns(
        n_assets=n_assets, n_periods=n_periods, crisis_at=350
    )

    # Build graph snapshots
    window = 40
    step = 5
    snapshots, timestamps, _ = build_synthetic_graphs(returns, window, step, threshold=0.25)
    n_snaps = len(snapshots)

    if verbose:
        print(f"Built {n_snaps} snapshots from {n_periods} periods of {n_assets} assets.")

    # Ground truth: wormhole labels
    # Pre-crisis and crisis periods are "true" wormhole periods
    true_crisis = []
    for ts in timestamps:
        # Crisis starts at 350, pre-crisis window at 320
        is_crisis = 320 <= ts <= 400
        true_crisis.append(1 if is_crisis else 0)

    # Initialize wormhole detector
    detector = WormholeDetector(contamination=0.1, window_size=15, z_score_threshold=2.0)

    # Training phase: use first 50% of snapshots
    train_cutoff = n_snaps // 2
    alarm_scores = []

    for t_idx, snap in enumerate(snapshots):
        ei = snap.edge_index.numpy()
        ew = snap.edge_attr[:, 0].numpy() if snap.edge_attr is not None else np.ones(ei.shape[1])
        edges = [(int(ei[0, e]), int(ei[1, e]), float(ew[e])) for e in range(ei.shape[1])]

        detector.update(edges, n_assets)

        if t_idx == train_cutoff:
            detector.fit()
            if verbose:
                print(f"Detector fitted on {t_idx} snapshots.")

        if t_idx >= train_cutoff:
            scores = detector.score(edges, n_assets)
            mean_score = float(np.mean(list(scores.values()))) if scores else 0.0
            alarm_scores.append(mean_score)
        else:
            alarm_scores.append(0.0)

    # Evaluate: classify each snapshot as crisis/not based on threshold
    alarm_arr = np.array(alarm_scores)
    true_arr = np.array(true_crisis)

    # Only evaluate on post-training snapshots
    post_train = alarm_arr[train_cutoff:]
    post_train_true = true_arr[train_cutoff:]

    threshold_vals = np.linspace(0, 1, 51)
    best_f1 = 0.0
    best_thresh = 0.5
    for th in threshold_vals:
        preds = (post_train >= th).astype(int)
        tp = ((preds == 1) & (post_train_true == 1)).sum()
        fp = ((preds == 1) & (post_train_true == 0)).sum()
        fn = ((preds == 0) & (post_train_true == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th

    preds_final = (post_train >= best_thresh).astype(int)
    tp = int(((preds_final == 1) & (post_train_true == 1)).sum())
    fp = int(((preds_final == 1) & (post_train_true == 0)).sum())
    fn = int(((preds_final == 0) & (post_train_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = best_f1

    # AUC-ROC
    from sklearn.metrics import roc_auc_score
    try:
        auc = float(roc_auc_score(post_train_true, post_train))
    except Exception:
        auc = 0.5

    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc,
        "best_threshold": best_thresh,
        "alarm_scores": alarm_scores,
        "true_labels": true_crisis,
        "n_snapshots": n_snaps,
        "runtime_s": time.time() - t0,
    }

    if verbose:
        print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}  AUC: {auc:.3f}")

    # Save results to CSV
    df = pd.DataFrame({
        "timestamp": timestamps,
        "alarm_score": alarm_scores,
        "true_crisis": true_crisis,
    })
    df.to_csv(os.path.join(output_dir, "wormhole_detection_results.csv"), index=False)

    # Plot
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(timestamps, alarm_scores, 'b-', label="Wormhole anomaly score")
        axes[0].axhline(best_thresh, color='r', linestyle='--', label=f"Threshold={best_thresh:.2f}")
        axes[0].fill_between(
            timestamps,
            0, 1,
            where=[tc == 1 for tc in true_crisis],
            alpha=0.2, color='red', label="True crisis"
        )
        axes[0].set_ylabel("Anomaly Score")
        axes[0].legend()
        axes[0].set_title(f"Wormhole Detection  AUC={auc:.3f}  F1={f1:.3f}")

        axes[1].plot(timestamps, [len(snap.edge_index[0]) for snap in snapshots], 'g-', label="Edge count")
        axes[1].fill_between(
            timestamps, 0,
            max(len(snap.edge_index[0]) for snap in snapshots),
            where=[tc == 1 for tc in true_crisis],
            alpha=0.2, color='red',
        )
        axes[1].set_xlabel("Time step")
        axes[1].set_ylabel("Edge Count")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "wormhole_detection.png"), dpi=150)
        plt.close()
        if verbose:
            print(f"Plot saved to {output_dir}/wormhole_detection.png")

    return results


# ── Experiment 2: Regime classification ──────────────────────────────────────

def run_regime_classification_experiment(
    output_dir: str = "outputs",
    n_assets: int = 20,
    n_periods: int = 600,
    n_regimes: int = 4,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Classify market regimes from graph topology.

    Compares:
    1. GraphRegimeDetector (topology features + K-means).
    2. Baseline: K-means on raw correlation matrix.

    Returns:
        Results dict with adjusted Rand index, accuracy, confusion matrix.
    """
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    if verbose:
        print("\n=== Experiment 2: Regime Classification ===")

    returns, true_regimes, _ = generate_synthetic_returns(
        n_assets=n_assets, n_periods=n_periods, n_regimes=n_regimes, seed=99
    )

    # Build snapshots
    window, step = 40, 5
    evo = GraphEvolution(returns, window=window, step=step, threshold=0.25)
    evo.run(verbose=verbose)

    # Align regime labels: each snapshot corresponds to end of window
    snap_regimes = [int(true_regimes[ts - 1]) for ts in evo.timestamps]

    # Compute Ricci curvature approximations
    ricci_curvs = [approximate_ricci_numpy(snap) for snap in evo.snapshots]
    evo.set_ricci_curvatures(ricci_curvs)

    # Split train/test
    n_snaps = len(evo.snapshots)
    train_n = int(n_snaps * 0.7)
    train_snaps = evo.snapshots[:train_n]
    test_snaps = evo.snapshots[train_n:]
    train_rc = ricci_curvs[:train_n]
    test_rc = ricci_curvs[train_n:]
    train_labels = snap_regimes[:train_n]
    test_labels = snap_regimes[train_n:]

    if verbose:
        print(f"Train: {train_n} snapshots, Test: {len(test_snaps)} snapshots")
        print(f"Regime distribution: {dict(zip(*np.unique(train_labels, return_counts=True)))}")

    # Fit GraphRegimeDetector
    detector = GraphRegimeDetector(n_regimes=n_regimes, n_init=5)
    detector.fit(train_snaps, train_rc)

    # Test predictions
    test_preds = [
        detector.predict(test_snaps[i], test_rc[i])
        for i in range(len(test_snaps))
    ]

    # Evaluate
    from sklearn.metrics import adjusted_rand_score, confusion_matrix, accuracy_score
    ari = float(adjusted_rand_score(test_labels, test_preds))

    # Map predicted labels to true labels (best permutation)
    # ARI handles permutation invariance
    best_accuracy = 0.0
    from itertools import permutations
    if n_regimes <= 5:
        for perm in permutations(range(n_regimes)):
            mapped = [perm[p] for p in test_preds]
            acc = float(accuracy_score(test_labels, mapped))
            best_accuracy = max(best_accuracy, acc)
    else:
        best_accuracy = float(accuracy_score(test_labels, test_preds))

    cm = confusion_matrix(test_labels, test_preds, labels=list(range(n_regimes)))

    # Baseline: K-means on mean return and volatility features
    from sklearn.cluster import KMeans
    baseline_features = np.column_stack([
        [snap.x[:, 0].mean().item() for snap in test_snaps],
        [snap.x[:, 1].mean().item() for snap in test_snaps],
        [snap.edge_index.size(1) for snap in test_snaps],
    ])
    kmeans_baseline = KMeans(n_clusters=n_regimes, n_init=10, random_state=42)
    # Fit on train features too
    train_features = np.column_stack([
        [snap.x[:, 0].mean().item() for snap in train_snaps],
        [snap.x[:, 1].mean().item() for snap in train_snaps],
        [snap.edge_index.size(1) for snap in train_snaps],
    ])
    kmeans_baseline.fit(train_features)
    baseline_preds = kmeans_baseline.predict(baseline_features)
    baseline_ari = float(adjusted_rand_score(test_labels, baseline_preds))

    results = {
        "ari": ari,
        "best_accuracy": best_accuracy,
        "baseline_ari": baseline_ari,
        "improvement_over_baseline": ari - baseline_ari,
        "n_train": train_n,
        "n_test": len(test_snaps),
        "confusion_matrix": cm.tolist(),
        "runtime_s": time.time() - t0,
    }

    if verbose:
        print(f"GraphRegimeDetector ARI: {ari:.3f}  Best Accuracy: {best_accuracy:.3f}")
        print(f"Baseline ARI: {baseline_ari:.3f}  Improvement: {ari - baseline_ari:+.3f}")

    # Save results
    pd.DataFrame({
        "test_true": test_labels,
        "test_pred": test_preds,
        "baseline_pred": list(baseline_preds),
    }).to_csv(os.path.join(output_dir, "regime_classification_results.csv"), index=False)

    # Plot confusion matrix
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        regime_names = [detector.get_regime_name(r) for r in range(n_regimes)]

        im = axes[0].imshow(cm, cmap="Blues")
        axes[0].set_xticks(range(n_regimes))
        axes[0].set_yticks(range(n_regimes))
        axes[0].set_xticklabels([f"P{i}" for i in range(n_regimes)], rotation=45)
        axes[0].set_yticklabels([f"T{i}" for i in range(n_regimes)])
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        axes[0].set_title(f"Regime Confusion Matrix\nARI={ari:.3f}")
        plt.colorbar(im, ax=axes[0])

        for r in range(n_regimes):
            for c in range(n_regimes):
                axes[0].text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=10)

        # Timeline of predicted regimes
        axes[1].scatter(range(len(test_labels)), test_labels, c=test_labels,
                        cmap="tab10", alpha=0.7, s=20, label="True")
        axes[1].scatter(range(len(test_preds)), [p + 0.1 for p in test_preds],
                        c=test_preds, cmap="tab10", marker='^', alpha=0.5, s=20, label="Pred")
        axes[1].set_xlabel("Test snapshot")
        axes[1].set_ylabel("Regime")
        axes[1].set_title("Predicted vs True Regimes")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "regime_classification.png"), dpi=150)
        plt.close()

    return results


# ── Experiment 3: Crisis prediction ──────────────────────────────────────────

def run_crisis_prediction_experiment(
    output_dir: str = "outputs",
    n_assets: int = 20,
    n_periods: int = 600,
    crisis_at: int = 420,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Predict crisis timing from Ricci curvature trajectory.

    Tests whether the combined alarm system (Ricci + regime transition +
    wormhole) can predict crisis onset with positive lead time.

    Metric: Mean lead time (steps before crisis at which alarm first fires).

    Returns:
        Results dict with lead_time, false_alarm_rate, detection_rate.
    """
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    if verbose:
        print("\n=== Experiment 3: Crisis Prediction ===")

    returns, regimes, crisis_mask = generate_synthetic_returns(
        n_assets=n_assets, n_periods=n_periods, crisis_at=crisis_at, seed=7
    )

    # Build graph snapshots
    window, step = 40, 5
    evo = GraphEvolution(returns, window=window, step=step, threshold=0.25)
    evo.run(verbose=False)
    n_snaps = len(evo.snapshots)

    # Compute approximate Ricci curvatures
    ricci_list = [approximate_ricci_numpy(snap) for snap in evo.snapshots]
    evo.set_ricci_curvatures(ricci_list)

    # Build graph features for regime detection
    regime_detector = GraphRegimeDetector(n_regimes=4, n_init=5)
    train_n = min(int(n_snaps * 0.5), (crisis_at - window) // step)
    train_n = max(train_n, 10)
    regime_detector.fit(evo.snapshots[:train_n], ricci_list[:train_n])

    # Align timestamps to regime labels
    snap_regimes = [int(regimes[min(ts - 1, n_periods - 1)]) for ts in evo.timestamps]

    # Initialize crisis early warning system
    # Crisis regime = regime most commonly observed during actual crisis
    crisis_snap_regimes = [
        snap_regimes[i] for i, ts in enumerate(evo.timestamps)
        if crisis_at <= ts <= crisis_at + 50
    ]
    crisis_regime = int(max(set(crisis_snap_regimes), key=crisis_snap_regimes.count)) \
        if crisis_snap_regimes else 3

    early_warning = CrisisEarlyWarning(
        n_regimes=4,
        crisis_regimes=[crisis_regime],
        ricci_weight=0.45,
        transition_weight=0.35,
        wormhole_weight=0.20,
        ph_delta=0.005,
        ph_lambda=20.0,
        ema_alpha=0.25,
    )

    # Run wormhole detector alongside
    wormhole_detector = WormholeDetector(contamination=0.1, window_size=12, z_score_threshold=2.5)

    alarm_scores = []
    ph_alarms = []

    for t_idx in range(n_snaps):
        snap = evo.snapshots[t_idx]
        rc = ricci_list[t_idx]

        # Wormhole score
        ei = snap.edge_index.numpy()
        ew = snap.edge_attr[:, 0].numpy() if snap.edge_attr is not None else np.ones(ei.shape[1])
        edges = [(int(ei[0, e]), int(ei[1, e]), float(ew[e])) for e in range(ei.shape[1])]
        wormhole_detector.update(edges, n_assets)
        if t_idx >= train_n:
            wormhole_detector.fit()
        wormhole_scores = wormhole_detector.score(edges, n_assets)
        wormhole_score = float(np.mean(list(wormhole_scores.values()))) if wormhole_scores else 0.0

        # Mean Ricci
        mean_rc = float(np.mean(rc)) if len(rc) > 0 else 0.0

        # Current regime
        pred_regime = regime_detector.predict(snap, rc)
        transition_probs = regime_detector.predict_proba(snap, rc)

        # Update early warning
        result = early_warning.update(
            mean_ricci=mean_rc,
            current_regime=pred_regime,
            transition_probs=transition_probs,
            wormhole_score=wormhole_score,
        )
        alarm_scores.append(result["alarm_score"])
        ph_alarms.append(result["ph_alarm"])

    # Evaluation: find first alarm before crisis
    alarm_arr = np.array(alarm_scores)
    crisis_start_snap = next(
        (i for i, ts in enumerate(evo.timestamps) if ts >= crisis_at),
        n_snaps - 1
    )
    crisis_end_snap = min(crisis_start_snap + 50 // step, n_snaps - 1)

    # True crisis snapshot range
    true_crisis_snaps = set(range(crisis_start_snap, crisis_end_snap + 1))

    # Find alarm threshold that maximizes detection - minimize false alarms
    alarm_threshold = 0.55
    first_alarm_before_crisis = None
    for i in range(crisis_start_snap - 1, max(0, crisis_start_snap - 30), -1):
        if alarm_arr[i] >= alarm_threshold:
            first_alarm_before_crisis = crisis_start_snap - i
            break

    # Count false alarms (alarms outside crisis window)
    non_crisis_alarms = sum(
        1 for i, a in enumerate(alarm_arr)
        if a >= alarm_threshold and i not in true_crisis_snaps
    )
    false_alarm_rate = non_crisis_alarms / max(n_snaps - len(true_crisis_snaps), 1)

    # Detection rate: fraction of crisis snapshots where alarm >= threshold
    crisis_alarm_scores = [alarm_arr[i] for i in true_crisis_snaps if i < len(alarm_arr)]
    detection_rate = sum(1 for s in crisis_alarm_scores if s >= alarm_threshold) / max(len(crisis_alarm_scores), 1)

    lead_time = first_alarm_before_crisis if first_alarm_before_crisis is not None else 0

    # Ricci trajectory comparison
    ricci_trajectory = [float(np.mean(rc)) if len(rc) > 0 else 0.0 for rc in ricci_list]

    results = {
        "lead_time_steps": lead_time,
        "lead_time_periods": lead_time * step,
        "detection_rate": detection_rate,
        "false_alarm_rate": false_alarm_rate,
        "crisis_start_snap": crisis_start_snap,
        "first_alarm_snap": crisis_start_snap - lead_time if lead_time > 0 else None,
        "mean_alarm_in_crisis": float(np.mean(crisis_alarm_scores)) if crisis_alarm_scores else 0.0,
        "runtime_s": time.time() - t0,
    }

    if verbose:
        print(f"Lead time: {lead_time} steps ({lead_time * step} periods)")
        print(f"Detection rate: {detection_rate:.3f}")
        print(f"False alarm rate: {false_alarm_rate:.4f}")

    # Save results
    pd.DataFrame({
        "snapshot": range(n_snaps),
        "timestamp": evo.timestamps,
        "alarm_score": alarm_scores,
        "ph_alarm": ph_alarms,
        "ricci_mean": ricci_trajectory,
        "snap_regime": snap_regimes,
    }).to_csv(os.path.join(output_dir, "crisis_prediction_results.csv"), index=False)

    # Plot
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Alarm score
        axes[0].plot(evo.timestamps, alarm_scores, 'b-', linewidth=1.5, label="Alarm Score")
        axes[0].axhline(alarm_threshold, color='orange', linestyle='--', label=f"Threshold={alarm_threshold}")
        axes[0].axvspan(
            evo.timestamps[crisis_start_snap],
            evo.timestamps[min(crisis_end_snap, len(evo.timestamps)-1)],
            alpha=0.2, color='red', label="Crisis period",
        )
        if first_alarm_before_crisis:
            axes[0].axvline(
                evo.timestamps[crisis_start_snap - first_alarm_before_crisis],
                color='purple', linestyle=':', label=f"First alarm (lead={lead_time} steps)"
            )
        axes[0].set_ylabel("Alarm Score")
        axes[0].legend(fontsize=8)
        axes[0].set_title("Crisis Early Warning Alarm")

        # Ricci trajectory
        axes[1].plot(evo.timestamps, ricci_trajectory, 'g-', linewidth=1.5, label="Mean Ricci Curvature")
        axes[1].axhline(0, color='gray', linestyle='-', alpha=0.5)
        axes[1].axvspan(
            evo.timestamps[crisis_start_snap],
            evo.timestamps[min(crisis_end_snap, len(evo.timestamps)-1)],
            alpha=0.2, color='red',
        )
        axes[1].set_ylabel("Mean Ricci")
        axes[1].legend()

        # Regime
        axes[2].scatter(evo.timestamps, snap_regimes, c=snap_regimes, cmap="tab10", s=20)
        axes[2].set_ylabel("Regime")
        axes[2].set_xlabel("Time step")
        axes[2].set_title("Detected Regimes")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "crisis_prediction.png"), dpi=150)
        plt.close()
        if verbose:
            print(f"Plot saved to {output_dir}/crisis_prediction.png")

    return results


# ── Experiment 4: Full benchmark ──────────────────────────────────────────────

def run_full_benchmark(
    output_dir: str = "outputs",
    n_assets: int = 15,
    n_periods: int = 400,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run all experiments and aggregate results into a benchmark report.

    Returns:
        Combined results dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("OMNI-GRAPH BENCHMARK SUITE")
        print(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"Assets: {n_assets}  Periods: {n_periods}")
        print("=" * 60)

    results = {}

    # Experiment 1: Wormhole detection
    results["wormhole"] = run_wormhole_detection_experiment(
        output_dir=output_dir,
        n_assets=n_assets,
        n_periods=n_periods,
        verbose=verbose,
    )

    # Experiment 2: Regime classification
    results["regime"] = run_regime_classification_experiment(
        output_dir=output_dir,
        n_assets=n_assets,
        n_periods=n_periods,
        verbose=verbose,
    )

    # Experiment 3: Crisis prediction
    crisis_at = int(n_periods * 0.75)
    results["crisis"] = run_crisis_prediction_experiment(
        output_dir=output_dir,
        n_assets=n_assets,
        n_periods=n_periods,
        crisis_at=crisis_at,
        verbose=verbose,
    )

    # Aggregate summary
    summary = {
        "wormhole_auc": results["wormhole"]["auc_roc"],
        "wormhole_f1": results["wormhole"]["f1"],
        "regime_ari": results["regime"]["ari"],
        "regime_accuracy": results["regime"]["best_accuracy"],
        "crisis_lead_time": results["crisis"]["lead_time_steps"],
        "crisis_detection_rate": results["crisis"]["detection_rate"],
        "crisis_false_alarm_rate": results["crisis"]["false_alarm_rate"],
        "total_runtime_s": sum(r.get("runtime_s", 0) for r in results.values()),
    }

    # Save summary
    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, "benchmark_summary.csv"), index=False
    )

    if verbose:
        print("\n=== BENCHMARK SUMMARY ===")
        for k, v in summary.items():
            print(f"  {k:<35}: {v:.4f}")

    return {"experiments": results, "summary": summary}


if __name__ == "__main__":
    run_full_benchmark(output_dir="outputs", n_assets=15, n_periods=400, verbose=True)
