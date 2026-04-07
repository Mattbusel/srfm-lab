# ml/training/feature_importance.py -- feature importance analysis for SRFM signal models
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IC (information coefficient) helper
# ---------------------------------------------------------------------------


def _rank_ic(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Spearman rank correlation between predictions and labels."""
    if len(y_pred) < 3:
        return 0.0
    corr, _ = stats.spearmanr(y_pred, y_true)
    return float(corr) if not np.isnan(corr) else 0.0


def _predict_safe(model: Any, X: np.ndarray) -> np.ndarray:
    """Call model.predict(X) or model(X) gracefully."""
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X), dtype=float)
    if callable(model):
        return np.asarray(model(X), dtype=float)
    raise TypeError(f"Cannot call model of type {type(model)}")


# ---------------------------------------------------------------------------
# Mutual information (discrete estimation)
# ---------------------------------------------------------------------------


def _discretize(x: np.ndarray, bins: int = 10) -> np.ndarray:
    """Bin a 1-D array into integer bin indices."""
    finite = x[np.isfinite(x)]
    if len(finite) == 0:
        return np.zeros_like(x, dtype=int)
    lo, hi = finite.min(), finite.max()
    if lo == hi:
        return np.zeros_like(x, dtype=int)
    edges = np.linspace(lo, hi, bins + 1)
    edges[-1] += 1e-12  # include right edge
    return np.digitize(x, edges) - 1


def _mutual_information_1d(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Compute MI(x; y) using discrete approximation."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 5:
        return 0.0

    xd = _discretize(x, bins)
    yd = _discretize(y, bins)

    n = len(x)
    # Joint distribution
    joint_counts: Dict[Tuple[int, int], int] = {}
    for xi, yi in zip(xd.tolist(), yd.tolist()):
        key = (int(xi), int(yi))
        joint_counts[key] = joint_counts.get(key, 0) + 1

    x_counts: Dict[int, int] = {}
    y_counts: Dict[int, int] = {}
    for xi, yi in zip(xd.tolist(), yd.tolist()):
        x_counts[int(xi)] = x_counts.get(int(xi), 0) + 1
        y_counts[int(yi)] = y_counts.get(int(yi), 0) + 1

    mi = 0.0
    for (xi, yi), cnt in joint_counts.items():
        p_xy = cnt / n
        p_x = x_counts[xi] / n
        p_y = y_counts[yi] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * np.log(p_xy / (p_x * p_y))

    return max(0.0, float(mi))


# ---------------------------------------------------------------------------
# FeatureImportanceAnalyzer
# ---------------------------------------------------------------------------


class FeatureImportanceAnalyzer:
    """
    Collection of feature importance methods for SRFM signal models.

    All methods return results as pandas Series or DataFrames indexed by
    feature name, sorted descending by importance score.

    Parameters
    ----------
    random_state:
        Seed for permutation randomness.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Permutation importance
    # ------------------------------------------------------------------

    def permutation_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_repeats: int = 10,
    ) -> pd.DataFrame:
        """
        Compute permutation feature importance by measuring IC degradation.

        For each feature, shuffle its values n_repeats times, measure the
        drop in Spearman IC, and report mean and std of the drop.

        Parameters
        ----------
        model:
            Fitted model with a predict() method.
        X:
            Feature matrix (n_samples x n_features).
        y:
            Target series of length n_samples.
        n_repeats:
            Number of permutation repetitions per feature.

        Returns
        -------
        pd.DataFrame
            Columns: feature, importance_mean, importance_std, base_ic
        """
        X_arr = X.values.astype(float)
        y_arr = y.values.astype(float)
        feature_names = list(X.columns)

        base_preds = _predict_safe(model, X_arr)
        base_ic = _rank_ic(base_preds, y_arr)

        records = []
        rng = np.random.default_rng(self.random_state)

        for i, feat in enumerate(feature_names):
            drops = []
            for _ in range(n_repeats):
                X_perm = X_arr.copy()
                perm_idx = rng.permutation(len(X_arr))
                X_perm[:, i] = X_arr[perm_idx, i]
                perm_preds = _predict_safe(model, X_perm)
                perm_ic = _rank_ic(perm_preds, y_arr)
                drops.append(base_ic - perm_ic)

            records.append({
                "feature": feat,
                "importance_mean": float(np.mean(drops)),
                "importance_std": float(np.std(drops)),
                "base_ic": base_ic,
            })

        df = pd.DataFrame(records)
        df.sort_values("importance_mean", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ------------------------------------------------------------------
    # SHAP approximation
    # ------------------------------------------------------------------

    def shap_values(
        self,
        model: Any,
        X: pd.DataFrame,
        method: str = "linear",
    ) -> np.ndarray:
        """
        Compute approximate SHAP values.

        For linear models: shap_i = w_i * (x_i - mean(x_i)) / std(predictions).
        For trees: recursively attribute variance reduction to splits.

        Parameters
        ----------
        model:
            Fitted model.  For "linear" method, model.coef_ must exist.
            For "tree" method, model must have tree_ attribute (sklearn DecisionTree).
        X:
            Feature matrix (n_samples x n_features).
        method:
            "linear" or "tree".

        Returns
        -------
        np.ndarray of shape (n_samples, n_features).
        """
        X_arr = X.values.astype(float)

        if method == "linear":
            return self._shap_linear(model, X_arr)
        elif method == "tree":
            return self._shap_tree(model, X_arr)
        else:
            raise ValueError(f"Unknown SHAP method: {method!r}. Use 'linear' or 'tree'.")

    def _shap_linear(self, model: Any, X_arr: np.ndarray) -> np.ndarray:
        """Linear SHAP: w_i * (x_i - x_bar) / std(preds)."""
        if not hasattr(model, "coef_"):
            # Fall back: estimate weights via correlation
            preds = _predict_safe(model, X_arr)
            shap = np.zeros_like(X_arr)
            std_p = np.std(preds) + 1e-12
            x_mean = X_arr.mean(axis=0)
            for i in range(X_arr.shape[1]):
                w_i, _ = stats.pearsonr(X_arr[:, i], preds)
                shap[:, i] = w_i * (X_arr[:, i] - x_mean[i]) / std_p
            return shap

        weights = np.asarray(model.coef_).ravel()
        if len(weights) != X_arr.shape[1]:
            weights = weights[: X_arr.shape[1]]

        preds = _predict_safe(model, X_arr)
        std_p = np.std(preds) + 1e-12
        x_mean = X_arr.mean(axis=0)
        shap = np.zeros_like(X_arr)
        for i, w in enumerate(weights):
            shap[:, i] = w * (X_arr[:, i] - x_mean[i]) / std_p
        return shap

    def _shap_tree(self, model: Any, X_arr: np.ndarray) -> np.ndarray:
        """
        Simple recursive SHAP for sklearn DecisionTreeRegressor/Classifier.

        Attributes prediction for each sample by traversing the decision path
        and allocating (node_value_right - node_value_left) to the split feature.
        """
        if not hasattr(model, "tree_"):
            raise AttributeError("Tree SHAP requires a sklearn DecisionTree model with .tree_")

        tree = model.tree_
        n_samples, n_features = X_arr.shape
        shap = np.zeros((n_samples, n_features))

        for s in range(n_samples):
            node = 0
            while tree.feature[node] != -2:  # -2 means leaf
                feat = tree.feature[node]
                threshold = tree.threshold[node]

                left_node = tree.children_left[node]
                right_node = tree.children_right[node]

                left_val = tree.value[left_node].ravel().mean()
                right_val = tree.value[right_node].ravel().mean()

                contribution = (right_val - left_val) * (
                    1 if X_arr[s, feat] > threshold else -1
                )
                shap[s, feat] += contribution

                if X_arr[s, feat] <= threshold:
                    node = left_node
                else:
                    node = right_node

        return shap

    def shap_summary(self, shap_values: np.ndarray, feature_names: List[str]) -> pd.Series:
        """Return mean absolute SHAP per feature as a sorted Series."""
        mean_abs = np.abs(shap_values).mean(axis=0)
        s = pd.Series(mean_abs, index=feature_names, name="mean_abs_shap")
        return s.sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Mutual information
    # ------------------------------------------------------------------

    def mutual_information(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        bins: int = 10,
    ) -> pd.Series:
        """
        Compute mutual information between each feature and the target.

        Uses discrete approximation: both X columns and y are binned into
        equal-width bins before computing MI.

        Returns
        -------
        pd.Series indexed by feature name, sorted descending.
        """
        y_arr = y.values.astype(float)
        mi_scores: Dict[str, float] = {}

        for col in X.columns:
            x_arr = X[col].values.astype(float)
            mi_scores[col] = _mutual_information_1d(x_arr, y_arr, bins=bins)

        result = pd.Series(mi_scores, name="mutual_information")
        return result.sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Correlation importance
    # ------------------------------------------------------------------

    def correlation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.Series:
        """
        Compute absolute Spearman correlation between each feature and target.

        Returns
        -------
        pd.Series indexed by feature name, sorted descending.
        """
        y_arr = y.values.astype(float)
        scores: Dict[str, float] = {}

        for col in X.columns:
            x_arr = X[col].values.astype(float)
            mask = np.isfinite(x_arr) & np.isfinite(y_arr)
            if mask.sum() < 3:
                scores[col] = 0.0
                continue
            corr, _ = stats.spearmanr(x_arr[mask], y_arr[mask])
            scores[col] = abs(float(corr)) if not np.isnan(corr) else 0.0

        result = pd.Series(scores, name="spearman_correlation")
        return result.sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Forward selection
    # ------------------------------------------------------------------

    def forward_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: Any,
        max_features: int = 20,
        fit_fn: Optional[Callable] = None,
    ) -> List[str]:
        """
        Greedy forward feature selection by IC improvement.

        At each step, adds the feature that most improves the IC when
        combined with the already-selected features.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Target.
        model:
            A model that can be re-fitted at each step.  Must implement
            fit(X, y) and predict(X), or a factory function can be
            supplied via fit_fn.
        max_features:
            Stop when this many features have been selected.
        fit_fn:
            Optional callable(X_train, y_train) -> fitted_model.
            If None, model.fit(X, y) is used.

        Returns
        -------
        List[str]
            Ordered list of selected feature names.
        """
        remaining = list(X.columns)
        selected: List[str] = []
        y_arr = y.values.astype(float)

        def _fit_and_ic(features: List[str]) -> float:
            Xs = X[features].values.astype(float)
            if fit_fn is not None:
                fitted = fit_fn(Xs, y_arr)
            else:
                model.fit(Xs, y_arr)
                fitted = model
            preds = _predict_safe(fitted, Xs)
            return _rank_ic(preds, y_arr)

        current_ic = 0.0

        for step in range(min(max_features, len(remaining))):
            best_feat: Optional[str] = None
            best_ic = current_ic

            for feat in remaining:
                candidate = selected + [feat]
                try:
                    ic = _fit_and_ic(candidate)
                except Exception:  # noqa: BLE001
                    ic = 0.0

                if ic > best_ic:
                    best_ic = ic
                    best_feat = feat

            if best_feat is None:
                # No improvement found
                break

            selected.append(best_feat)
            remaining.remove(best_feat)
            current_ic = best_ic
            logger.debug("Forward selection step %d: added %s, IC=%.4f", step + 1, best_feat, best_ic)

        return selected

    # ------------------------------------------------------------------
    # Backward elimination
    # ------------------------------------------------------------------

    def backward_elimination(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: Any,
        threshold: float = 0.01,
        fit_fn: Optional[Callable] = None,
    ) -> List[str]:
        """
        Greedy backward feature elimination by IC degradation.

        At each step, removes the feature whose removal causes the smallest
        drop in IC, provided that drop is less than `threshold`.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Target.
        model:
            Fitted model with fit(X, y) and predict(X).
        threshold:
            Minimum IC drop to stop elimination (i.e., removing a feature
            that would degrade IC by more than threshold is not allowed).
        fit_fn:
            Optional fit factory callable.

        Returns
        -------
        List[str]
            Features remaining after elimination.
        """
        selected = list(X.columns)
        y_arr = y.values.astype(float)

        def _fit_and_ic(features: List[str]) -> float:
            Xs = X[features].values.astype(float)
            if fit_fn is not None:
                fitted = fit_fn(Xs, y_arr)
            else:
                model.fit(Xs, y_arr)
                fitted = model
            preds = _predict_safe(fitted, Xs)
            return _rank_ic(preds, y_arr)

        # Compute baseline IC with all features
        try:
            base_ic = _fit_and_ic(selected)
        except Exception:  # noqa: BLE001
            return selected

        while len(selected) > 1:
            worst_feat: Optional[str] = None
            worst_drop = float("inf")

            for feat in selected:
                candidate = [f for f in selected if f != feat]
                try:
                    ic = _fit_and_ic(candidate)
                except Exception:  # noqa: BLE001
                    ic = base_ic

                drop = base_ic - ic
                if drop < worst_drop:
                    worst_drop = drop
                    worst_feat = feat

            if worst_drop > threshold or worst_feat is None:
                break

            selected.remove(worst_feat)
            base_ic = _fit_and_ic(selected)
            logger.debug(
                "Backward elimination: removed %s, IC=%.4f (drop was %.4f)",
                worst_feat,
                base_ic,
                worst_drop,
            )

        return selected


# ---------------------------------------------------------------------------
# FeatureImportanceReport
# ---------------------------------------------------------------------------

# Feature category mappings for SRFM signals
_CATEGORY_PREFIXES: Dict[str, List[str]] = {
    "technical": ["mom_", "rsi_", "macd_", "bb_", "atr_", "ema_", "sma_", "vol_", "vwap_"],
    "microstructure": ["ofi_", "bid_", "ask_", "spread_", "depth_", "imbalance_", "trade_"],
    "regime": ["regime_", "hurst_", "BH_", "markov_", "trend_"],
    "on_chain": ["btc_", "eth_", "miner_", "exchange_flow_", "active_addr_", "nvt_", "funding_"],
    "macro": ["dxy_", "vix_", "rates_", "inflation_", "gdp_"],
}


def _classify_feature(name: str) -> str:
    lower = name.lower()
    for category, prefixes in _CATEGORY_PREFIXES.items():
        for prefix in prefixes:
            if lower.startswith(prefix):
                return category
    return "other"


@dataclass
class FeatureImportanceReport:
    """
    Generates a formatted report from feature importance scores.

    Parameters
    ----------
    importance_scores:
        pd.Series indexed by feature name with importance values.
    feature_names:
        Explicit list of all feature names (optional; inferred from scores).
    correlation_matrix:
        Optional correlation matrix for redundancy detection.
    redundancy_threshold:
        Features with abs(correlation) above this threshold are flagged.
    """

    importance_scores: pd.Series
    feature_names: Optional[List[str]] = None
    correlation_matrix: Optional[pd.DataFrame] = None
    redundancy_threshold: float = 0.9

    def __post_init__(self) -> None:
        if self.feature_names is None:
            self.feature_names = list(self.importance_scores.index)

    def redundant_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Identify pairs of features with abs(correlation) > redundancy_threshold.

        Returns
        -------
        List of (feature_a, feature_b, correlation) tuples sorted by abs correlation.
        """
        if self.correlation_matrix is None:
            return []

        pairs: List[Tuple[str, str, float]] = []
        cols = list(self.correlation_matrix.columns)

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr = float(self.correlation_matrix.iloc[i, j])
                if abs(corr) > self.redundancy_threshold:
                    pairs.append((cols[i], cols[j], corr))

        pairs.sort(key=lambda t: abs(t[2]), reverse=True)
        return pairs

    def group_by_category(self) -> Dict[str, pd.Series]:
        """Group importance scores by feature category."""
        groups: Dict[str, Dict[str, float]] = {}
        for feat, score in self.importance_scores.items():
            cat = _classify_feature(str(feat))
            groups.setdefault(cat, {})[str(feat)] = float(score)

        return {
            cat: pd.Series(scores).sort_values(ascending=False)
            for cat, scores in groups.items()
        }

    def top_k(self, k: int = 20) -> pd.Series:
        """Return the top-k features by importance."""
        return self.importance_scores.head(k)

    def to_markdown(self, top_k: int = 30, title: str = "Feature Importance Report") -> str:
        """
        Generate a markdown table of feature importance results.

        Includes category grouping and redundancy warnings.
        """
        lines: List[str] = []
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"Total features analyzed: {len(self.importance_scores)}")
        lines.append("")

        # Top features table
        lines.append(f"## Top {top_k} Features")
        lines.append("")
        lines.append("| Rank | Feature | Category | Importance |")
        lines.append("|------|---------|----------|-----------|")

        for rank, (feat, score) in enumerate(
            self.importance_scores.head(top_k).items(), start=1
        ):
            cat = _classify_feature(str(feat))
            lines.append(f"| {rank} | {feat} | {cat} | {score:.6f} |")

        lines.append("")

        # Category summaries
        lines.append("## Category Summary")
        lines.append("")
        lines.append("| Category | Features | Total Importance | Mean Importance |")
        lines.append("|----------|----------|-----------------|-----------------|")

        for cat, series in sorted(self.group_by_category().items()):
            n = len(series)
            total = series.sum()
            mean = series.mean()
            lines.append(f"| {cat} | {n} | {total:.4f} | {mean:.4f} |")

        lines.append("")

        # Redundancy warnings
        pairs = self.redundant_pairs()
        if pairs:
            lines.append("## Redundant Feature Pairs (|correlation| > "
                         f"{self.redundancy_threshold:.2f})")
            lines.append("")
            lines.append("| Feature A | Feature B | Correlation |")
            lines.append("|-----------|-----------|-------------|")
            for fa, fb, corr in pairs[:20]:
                lines.append(f"| {fa} | {fb} | {corr:.4f} |")
            lines.append("")

        return "\n".join(lines)

    def suggested_drops(self) -> List[str]:
        """
        Suggest features to drop: below-median importance among redundant pairs.

        For each redundant pair, suggest dropping the one with lower importance.
        """
        pairs = self.redundant_pairs()
        to_drop: set = set()

        for fa, fb, _ in pairs:
            score_a = self.importance_scores.get(fa, 0.0)
            score_b = self.importance_scores.get(fb, 0.0)
            worse = fb if score_a >= score_b else fa
            to_drop.add(worse)

        return sorted(to_drop)
