"""
idea-engine/ingestion/miners/regime_cluster_miner.py
─────────────────────────────────────────────────────
Clusters regime feature vectors from regime_log and identifies clusters
where trade performance is significantly different from the baseline.

Method
──────
  1. Load regime_log (from LiveTradeData.regime_log).
  2. Extract feature vectors: [d_bh_mass, h_bh_mass, m15_bh_mass,
     tf_score, atr, garch_vol, ou_zscore].
  3. Standardise features (RobustScaler).
  4. Run HDBSCAN (or k-means elbow fallback if hdbscan not installed).
  5. For each regime row, find the nearest trade (by timestamp).
  6. Per cluster: compute PnL stats, compare against baseline with
     Mann-Whitney U, compute Cohen's d / Cliff's delta.
  7. Return significant clusters as MinedPattern objects.
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from ..config import (
    MIN_GROUP_SAMPLE,
    REGIME_FEATURE_COLS,
    REGIME_HDBSCAN_MIN_CLUSTER_SIZE,
    REGIME_HDBSCAN_MIN_SAMPLES,
    REGIME_KMEANS_K_RANGE,
    RAW_P_VALUE_THRESHOLD,
    MIN_EFFECT_SIZE,
)
from ..types import EffectSizeType, LiveTradeData, MinedPattern, PatternStatus, PatternType

logger = logging.getLogger(__name__)

_HAS_HDBSCAN = importlib.util.find_spec("hdbscan") is not None


# ── statistics helpers ────────────────────────────────────────────────────────

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta non-parametric effect size."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 0.0
    dominance = sum(1 if xi > xj else (-1 if xi < xj else 0) for xi in a for xj in b)
    return float(dominance / (n1 * n2))


def _bh_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(range(n), key=lambda i: p_values[i])
    reject  = [False] * n
    for rank, idx in enumerate(indexed, start=1):
        if p_values[idx] <= alpha * rank / n:
            reject[idx] = True
    return reject


# ── feature engineering ───────────────────────────────────────────────────────

def _build_feature_matrix(
    regime_log: pd.DataFrame,
    feature_cols: List[str] = REGIME_FEATURE_COLS,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract and standardise feature vectors from regime_log.

    Returns (X_scaled, used_cols) where X_scaled is (n_samples, n_features).
    """
    available = [c for c in feature_cols if c in regime_log.columns]
    if not available:
        raise ValueError(f"No feature columns found in regime_log. Expected any of: {feature_cols}")

    X = regime_log[available].astype(float).values

    # Replace NaN with column median
    for j in range(X.shape[1]):
        col_vals = X[:, j]
        nan_mask = np.isnan(col_vals)
        if nan_mask.any():
            median = np.nanmedian(col_vals)
            X[nan_mask, j] = median if not np.isnan(median) else 0.0

    # RobustScaler: subtract median, divide by IQR
    medians = np.median(X, axis=0)
    q25     = np.percentile(X, 25, axis=0)
    q75     = np.percentile(X, 75, axis=0)
    iqr     = q75 - q25
    iqr[iqr == 0] = 1.0
    X_scaled = (X - medians) / iqr

    return X_scaled, available


# ── clustering ────────────────────────────────────────────────────────────────

def _cluster_hdbscan(X: np.ndarray) -> np.ndarray:
    import hdbscan  # type: ignore
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=REGIME_HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=REGIME_HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info("HDBSCAN: %d clusters (noise=%d)", n_clusters, (labels == -1).sum())
    return labels


def _cluster_kmeans(X: np.ndarray, k_range: Tuple[int, int] = REGIME_KMEANS_K_RANGE) -> np.ndarray:
    from sklearn.cluster import KMeans  # type: ignore

    min_k, max_k = k_range
    max_k = min(max_k, len(X) // 10, 10)
    max_k = max(max_k, min_k)

    best_k, best_inertia, best_labels = min_k, float("inf"), None
    inertias = []

    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = km.fit_predict(X)
        inertias.append(km.inertia_)
        if k == min_k:
            best_k, best_inertia, best_labels = k, km.inertia_, labs
        else:
            # Elbow: improvement over previous
            improvement = (inertias[-2] - km.inertia_) / max(inertias[0], 1e-9)
            if improvement > 0.10:
                best_k, best_inertia, best_labels = k, km.inertia_, labs

    logger.info("k-means: chose k=%d", best_k)
    return best_labels if best_labels is not None else np.zeros(len(X), dtype=int)


def _cluster(X: np.ndarray) -> np.ndarray:
    if _HAS_HDBSCAN:
        try:
            return _cluster_hdbscan(X)
        except Exception as exc:
            logger.warning("HDBSCAN failed (%s), falling back to k-means", exc)
    try:
        return _cluster_kmeans(X)
    except Exception as exc:
        logger.warning("k-means also failed (%s), assigning all to cluster 0", exc)
        return np.zeros(len(X), dtype=int)


# ── trade–regime joining ──────────────────────────────────────────────────────

def _join_trades_to_regimes(
    trades: pd.DataFrame,
    regime_log: pd.DataFrame,
    cluster_labels: np.ndarray,
) -> pd.DataFrame:
    """
    Assign each trade a cluster label by nearest-prior regime timestamp.

    Returns trades with a new column 'cluster_label'.
    """
    trades = trades.copy()
    regime_log = regime_log.copy()

    ts_col = "ts" if "ts" in trades.columns else "exit_time"
    if ts_col not in trades.columns:
        logger.warning("No timestamp column in trades for regime join")
        trades["cluster_label"] = -1
        return trades

    trades["_ts"]  = pd.to_datetime(trades[ts_col], errors="coerce")
    regime_log["_ts"] = pd.to_datetime(regime_log["ts"], errors="coerce")
    regime_log     = regime_log.sort_values("_ts").reset_index(drop=True)
    regime_log["cluster_label"] = cluster_labels

    # Merge-asof: assign each trade the most recent regime label
    trades_sorted = trades.sort_values("_ts").reset_index(drop=False)
    merged = pd.merge_asof(
        trades_sorted,
        regime_log[["_ts", "cluster_label"]],
        on="_ts",
        direction="backward",
        tolerance=pd.Timedelta("6H"),
    )
    merged = merged.sort_values("index").set_index("index")
    trades["cluster_label"] = merged["cluster_label"].values
    trades.drop(columns=["_ts"], inplace=True, errors="ignore")
    return trades


# ── pattern extraction ────────────────────────────────────────────────────────

def _build_cluster_patterns(
    trades: pd.DataFrame,
    feature_cols: List[str],
    regime_log: pd.DataFrame,
    cluster_labels: np.ndarray,
    source: str,
    alpha: float = RAW_P_VALUE_THRESHOLD,
    min_effect: float = MIN_EFFECT_SIZE,
) -> List[MinedPattern]:
    patterns: List[MinedPattern] = []

    all_pnl = trades["pnl"].dropna().values
    if len(all_pnl) < MIN_GROUP_SAMPLE:
        return patterns

    unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]
    p_vals: List[float] = []
    candidates = []

    for c in unique_clusters:
        mask   = trades["cluster_label"] == c
        grp    = trades[mask]["pnl"].dropna().values
        if len(grp) < MIN_GROUP_SAMPLE:
            continue
        base   = trades[~mask]["pnl"].dropna().values
        if len(base) < MIN_GROUP_SAMPLE:
            continue
        try:
            _, p = mannwhitneyu(grp, base, alternative="two-sided")
        except Exception:
            p = 1.0
        p_vals.append(float(p))
        candidates.append((c, grp, base, float(p)))

    if not candidates:
        return patterns

    # BH correction
    rejected = _bh_correction([c[3] for c in candidates], alpha=alpha)

    for (c, grp, base, p), rej in zip(candidates, rejected):
        d      = _cohens_d(grp, base)
        delta  = _cliffs_delta(grp, base)
        effect = abs(delta)
        if not rej and effect < min_effect:
            continue

        # Compute representative feature centroid for this cluster
        cluster_rows = regime_log[cluster_labels == c]
        centroid: Dict[str, float] = {}
        for col in feature_cols:
            if col in cluster_rows.columns:
                centroid[col] = float(cluster_rows[col].median())

        win_r = float((grp > 0).sum() / len(grp))
        pf_g  = grp[grp > 0].sum()
        pf_l  = abs(grp[grp < 0].sum())
        pf    = float(pf_g / pf_l) if pf_l > 0 else None

        patterns.append(MinedPattern(
            source           = source,
            miner            = "RegimeClusterMiner",
            pattern_type     = PatternType.REGIME_CLUSTER,
            label            = f"Regime cluster {c}: anomalous PnL",
            description      = (
                f"Cluster {c} ({len(grp)} trades): mean PnL={grp.mean():.2f} "
                f"vs baseline={base.mean():.2f}; "
                f"Cliff's δ={delta:.3f}, Cohen's d={d:.3f}, p={p:.4f}"
            ),
            feature_dict     = {"cluster_id": int(c), "centroid": centroid},
            sample_size      = int(len(grp)),
            p_value          = p,
            effect_size      = effect,
            effect_size_type = EffectSizeType.CLIFFS_DELTA,
            win_rate         = win_r,
            avg_pnl          = float(grp.mean()),
            avg_pnl_baseline = float(base.mean()),
            profit_factor    = pf,
            status           = PatternStatus.NEW,
            tags             = ["regime", "cluster", f"cluster_{c}"],
            raw_group        = pd.Series(grp),
            raw_baseline     = pd.Series(base),
        ))

    return patterns


# ── public API ────────────────────────────────────────────────────────────────

class RegimeClusterMiner:
    """Finds regime clusters with anomalous trade performance."""

    def __init__(
        self,
        source:         str   = "live",
        feature_cols:   List[str] = REGIME_FEATURE_COLS,
        alpha:          float = RAW_P_VALUE_THRESHOLD,
        min_effect:     float = MIN_EFFECT_SIZE,
    ):
        self.source       = source
        self.feature_cols = feature_cols
        self.alpha        = alpha
        self.min_effect   = min_effect

    def mine(self, live_data: LiveTradeData) -> List[MinedPattern]:
        """
        Run regime cluster mining on a LiveTradeData object.

        Returns
        -------
        List[MinedPattern]
        """
        if live_data.regime_log is None or live_data.regime_log.empty:
            logger.warning("RegimeClusterMiner: no regime_log data")
            return []
        if live_data.trades is None or live_data.trades.empty:
            logger.warning("RegimeClusterMiner: no trades data")
            return []
        if "pnl" not in live_data.trades.columns:
            logger.warning("RegimeClusterMiner: trades missing 'pnl' column")
            return []

        logger.info("RegimeClusterMiner: building feature matrix …")
        try:
            X, used_cols = _build_feature_matrix(live_data.regime_log, self.feature_cols)
        except ValueError as exc:
            logger.error("RegimeClusterMiner feature build failed: %s", exc)
            return []

        logger.info("RegimeClusterMiner: clustering %d regime rows with %d features …", *X.shape)
        labels = _cluster(X)

        logger.info("RegimeClusterMiner: joining trades to regime clusters …")
        trades_labelled = _join_trades_to_regimes(
            live_data.trades,
            live_data.regime_log,
            labels,
        )

        logger.info("RegimeClusterMiner: extracting patterns …")
        patterns = _build_cluster_patterns(
            trades_labelled,
            used_cols,
            live_data.regime_log,
            labels,
            source    = self.source,
            alpha     = self.alpha,
            min_effect = self.min_effect,
        )

        logger.info("RegimeClusterMiner produced %d pattern(s)", len(patterns))
        return patterns


def mine_regime_clusters(live_data: LiveTradeData, source: str = "live", **kwargs) -> List[MinedPattern]:
    """Shortcut function."""
    return RegimeClusterMiner(source=source, **kwargs).mine(live_data)
