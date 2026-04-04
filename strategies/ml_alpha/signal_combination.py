"""
Signal combination methods for ML alpha signals.

Provides equal-weight, IC-weighted, PCA, and hierarchical combination
of multiple alpha signals into a single composite signal.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _align_signals(signals: List[pd.Series]) -> pd.DataFrame:
    """Align a list of signals on a common index, forward-filling gaps."""
    df = pd.concat(signals, axis=1)
    df.columns = [f"s{i}" for i in range(len(signals))]
    return df


def _rolling_ic(signal: pd.Series, forward_return: pd.Series, window: int = 60) -> pd.Series:
    """Rolling Spearman IC between a signal and forward returns."""
    ic_vals = []
    idx = signal.index
    for i in range(window, len(idx)):
        s_window = signal.iloc[i - window:i].dropna()
        r_window = forward_return.reindex(s_window.index).dropna()
        common = s_window.index.intersection(r_window.index)
        if len(common) < 10:
            ic_vals.append(np.nan)
            continue
        rho, _ = spearmanr(s_window.loc[common], r_window.loc[common])
        ic_vals.append(rho)
    return pd.Series(ic_vals, index=idx[window:], name=signal.name)


def _shrink_to_identity(cov: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    """Ledoit-Wolf style shrinkage toward scaled identity."""
    n = cov.shape[0]
    mu = np.trace(cov) / n
    return (1 - shrinkage) * cov + shrinkage * mu * np.eye(n)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SignalCombiner:
    """
    Combine multiple alpha signals into a single composite signal using
    various aggregation methods.

    Parameters
    ----------
    normalize_inputs : bool
        If True, z-score each input signal before combining.
    clip_z : float
        Clip input signals at +/- clip_z standard deviations.
    min_signals : int
        Minimum number of non-NaN signals required to produce an output.
    """

    def __init__(
        self,
        normalize_inputs: bool = True,
        clip_z: float = 3.0,
        min_signals: int = 1,
    ) -> None:
        self.normalize_inputs = normalize_inputs
        self.clip_z = clip_z
        self.min_signals = min_signals

    # ------------------------------------------------------------------
    # Internal preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, signals: List[pd.Series]) -> pd.DataFrame:
        """Align, optionally z-score, and clip signals."""
        df = _align_signals(signals)
        if self.normalize_inputs:
            mu = df.mean()
            sd = df.std().replace(0, np.nan)
            df = (df - mu) / sd
        if self.clip_z > 0:
            df = df.clip(-self.clip_z, self.clip_z)
        return df

    def _mask_min(self, composite: pd.Series, df: pd.DataFrame) -> pd.Series:
        """Set output to NaN where fewer than min_signals are available."""
        valid = df.notna().sum(axis=1)
        composite[valid < self.min_signals] = np.nan
        return composite

    # ------------------------------------------------------------------
    # Equal-weight combination
    # ------------------------------------------------------------------

    def equal_weight(self, signals: List[pd.Series]) -> pd.Series:
        """
        Combine signals by simple equal-weight average.

        Parameters
        ----------
        signals : list of pd.Series
            Alpha signals to combine.

        Returns
        -------
        pd.Series
            Equal-weighted composite signal.
        """
        df = self._preprocess(signals)
        composite = df.mean(axis=1, skipna=True)
        composite = self._mask_min(composite, df)
        composite.name = "equal_weight"
        return composite

    # ------------------------------------------------------------------
    # IC-weighted combination
    # ------------------------------------------------------------------

    def ic_weighted(
        self,
        signals: List[pd.Series],
        ic_history: Optional[List[pd.Series]] = None,
        ic_window: int = 60,
        forward_returns: Optional[pd.Series] = None,
        floor_weight: float = 0.0,
    ) -> pd.Series:
        """
        Combine signals using time-varying IC weights.

        If ic_history is provided it should be a list of pre-computed IC
        series (one per signal).  Otherwise forward_returns must be given
        and rolling IC is computed internally.

        Weights at each date are proportional to the trailing mean IC of each
        signal, clipped at floor_weight (default 0 – signals with negative IC
        receive zero weight).

        Parameters
        ----------
        signals : list of pd.Series
        ic_history : list of pd.Series, optional
            Pre-computed IC series aligned to the same index as signals.
        ic_window : int
            Window for rolling IC computation (ignored when ic_history given).
        forward_returns : pd.Series, optional
            Used when ic_history is None to compute rolling IC internally.
        floor_weight : float
            Minimum IC weight; set > 0 to always give some weight to every
            signal.

        Returns
        -------
        pd.Series
            IC-weighted composite signal.
        """
        df = self._preprocess(signals)

        if ic_history is None:
            if forward_returns is None:
                warnings.warn(
                    "ic_weighted: neither ic_history nor forward_returns "
                    "provided; falling back to equal-weight.",
                    stacklevel=2,
                )
                return self.equal_weight(signals)
            ic_history = [
                _rolling_ic(df.iloc[:, i], forward_returns, window=ic_window)
                for i in range(df.shape[1])
            ]

        ic_df = pd.concat(ic_history, axis=1).reindex(df.index).ffill()
        ic_df.columns = df.columns

        # Use trailing mean IC as weight
        ic_mean = ic_df.rolling(ic_window, min_periods=1).mean()
        ic_mean = ic_mean.clip(lower=floor_weight)  # floor at 0 or floor_weight
        weight_sum = ic_mean.sum(axis=1).replace(0, np.nan)

        composite_vals = []
        for date in df.index:
            row = df.loc[date]
            w = ic_mean.loc[date] if date in ic_mean.index else pd.Series(np.nan, index=df.columns)
            total_w = weight_sum.loc[date] if date in weight_sum.index else np.nan
            valid = row.notna() & w.notna() & (w > 0)
            if valid.sum() < self.min_signals or np.isnan(total_w) or total_w == 0:
                composite_vals.append(np.nan)
            else:
                composite_vals.append((row[valid] * w[valid]).sum() / w[valid].sum())

        composite = pd.Series(composite_vals, index=df.index, name="ic_weighted")
        return composite

    # ------------------------------------------------------------------
    # PCA combination
    # ------------------------------------------------------------------

    def pca_combine(
        self,
        signals: List[pd.Series],
        n_components: int = 1,
        refit_window: Optional[int] = None,
        min_fit_obs: int = 60,
    ) -> pd.Series:
        """
        Combine signals using the first principal component of the signal
        matrix, optionally refitting PCA on a rolling basis.

        Parameters
        ----------
        signals : list of pd.Series
        n_components : int
            Number of PCs to retain; output is the first PC projection.
        refit_window : int or None
            If given, refit PCA every refit_window periods.  Expanding window
            is used up to min_fit_obs, then rolling window of refit_window.
        min_fit_obs : int
            Minimum observations before emitting a signal.

        Returns
        -------
        pd.Series
            First-PC projection of the signal matrix.
        """
        df = self._preprocess(signals)
        n_signals = df.shape[1]
        n_components = min(n_components, n_signals)

        if refit_window is None:
            # Single PCA fit on full data (in-sample; use for research only)
            full = df.dropna(how="any")
            if len(full) < min_fit_obs:
                return pd.Series(np.nan, index=df.index, name="pca_combine")
            scaler = StandardScaler()
            X = scaler.fit_transform(full.values)
            pca = PCA(n_components=n_components)
            pca.fit(X)
            # Sign convention: largest absolute loading positive
            loadings = pca.components_[0]
            if np.abs(loadings).argmax() != loadings.argmax():
                loadings = -loadings

            composite_vals = []
            for i, date in enumerate(df.index):
                row = df.loc[date]
                if row.isna().any():
                    composite_vals.append(np.nan)
                else:
                    x_scaled = (row.values - scaler.mean_) / (scaler.scale_ + 1e-12)
                    composite_vals.append(float(x_scaled @ loadings))

            return pd.Series(composite_vals, index=df.index, name="pca_combine")

        # Rolling refit
        composite_vals = []
        loadings_cache: Optional[np.ndarray] = None
        scaler_cache: Optional[StandardScaler] = None
        last_fit = -refit_window  # force immediate fit

        for i, date in enumerate(df.index):
            # Determine training slice
            train_end = i
            train_start = max(0, train_end - refit_window) if i >= min_fit_obs else 0
            fit_needed = (i - last_fit) >= refit_window or loadings_cache is None

            if fit_needed and i >= min_fit_obs:
                train_df = df.iloc[train_start:train_end].dropna(how="any")
                if len(train_df) >= min_fit_obs // 2:
                    scaler = StandardScaler()
                    X = scaler.fit_transform(train_df.values)
                    pca = PCA(n_components=n_components)
                    pca.fit(X)
                    new_loadings = pca.components_[0].copy()
                    # Align sign with previous loadings if possible
                    if loadings_cache is not None:
                        if np.dot(new_loadings, loadings_cache) < 0:
                            new_loadings = -new_loadings
                    else:
                        # Orient so max-magnitude loading is positive
                        if np.abs(new_loadings).argmax() != new_loadings.argmax():
                            new_loadings = -new_loadings
                    loadings_cache = new_loadings
                    scaler_cache = scaler
                    last_fit = i

            row = df.loc[date]
            if row.isna().any() or loadings_cache is None or scaler_cache is None:
                composite_vals.append(np.nan)
            else:
                x_scaled = (row.values - scaler_cache.mean_) / (scaler_cache.scale_ + 1e-12)
                composite_vals.append(float(x_scaled @ loadings_cache))

        composite = pd.Series(composite_vals, index=df.index, name="pca_combine")
        return composite

    # ------------------------------------------------------------------
    # Hierarchical combination
    # ------------------------------------------------------------------

    def hierarchical_combine(
        self,
        signals: List[pd.Series],
        correlation_matrix: Optional[np.ndarray] = None,
        method: str = "ward",
        n_clusters: Optional[int] = None,
        intra_cluster_weight: str = "equal",
        shrinkage: float = 0.1,
    ) -> pd.Series:
        """
        Combine signals using hierarchical risk parity / clustering.

        Signals are clustered by their pairwise correlation (or a supplied
        correlation matrix).  Within each cluster, signals are averaged.
        Cluster composites are then averaged with equal weight across clusters
        (giving each cluster the same influence regardless of how many signals
        it contains).

        Parameters
        ----------
        signals : list of pd.Series
        correlation_matrix : np.ndarray, optional
            (n_signals x n_signals) correlation matrix.  Computed from data if
            not provided.
        method : str
            Linkage method: 'ward', 'complete', 'average', 'single'.
        n_clusters : int or None
            Number of clusters.  If None, set to max(2, n_signals // 3).
        intra_cluster_weight : str
            'equal' — equal weight within cluster.
            'iv'    — inverse-variance weight within cluster.
        shrinkage : float
            Shrinkage applied to estimated correlation matrix.

        Returns
        -------
        pd.Series
            Hierarchically combined composite signal.
        """
        df = self._preprocess(signals)
        n = df.shape[1]

        if n == 1:
            s = df.iloc[:, 0].copy()
            s.name = "hierarchical_combine"
            return s

        if n_clusters is None:
            n_clusters = max(2, n // 3)
        n_clusters = min(n_clusters, n)

        # Build correlation matrix
        if correlation_matrix is not None:
            corr = np.array(correlation_matrix)
        else:
            clean = df.dropna(how="any")
            if len(clean) < 20:
                warnings.warn(
                    "hierarchical_combine: insufficient data for correlation "
                    "estimation; falling back to equal-weight.",
                    stacklevel=2,
                )
                return self.equal_weight(signals)
            corr = clean.corr().values
            # Shrink
            corr = _shrink_to_identity(corr, shrinkage)
            # Force symmetry and clip to [-1, 1]
            corr = (corr + corr.T) / 2
            np.fill_diagonal(corr, 1.0)
            corr = np.clip(corr, -1.0, 1.0)

        # Distance matrix: d = sqrt(0.5 * (1 - rho))
        dist = np.sqrt(np.maximum(0, 0.5 * (1.0 - corr)))
        np.fill_diagonal(dist, 0.0)

        try:
            condensed = squareform(dist, checks=False)
            Z = linkage(condensed, method=method)
            labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        except Exception:
            return self.equal_weight(signals)

        # Build cluster composites then average clusters
        cluster_ids = np.unique(labels)
        cluster_composites = []

        for cid in cluster_ids:
            members = [i for i in range(n) if labels[i] == cid]
            sub = df.iloc[:, members]

            if intra_cluster_weight == "iv":
                vols = sub.std()
                iv_w = (1.0 / vols.replace(0, np.nan)).fillna(0)
                total_w = iv_w.sum()
                if total_w == 0:
                    clust_comp = sub.mean(axis=1, skipna=True)
                else:
                    clust_comp = (sub * iv_w / total_w).sum(axis=1, skipna=True)
            else:
                clust_comp = sub.mean(axis=1, skipna=True)

            cluster_composites.append(clust_comp)

        cluster_df = pd.concat(cluster_composites, axis=1)
        composite = cluster_df.mean(axis=1, skipna=True)

        # Mask where insufficient original signals
        valid = df.notna().sum(axis=1)
        composite[valid < self.min_signals] = np.nan
        composite.name = "hierarchical_combine"
        return composite

    # ------------------------------------------------------------------
    # Convenience: combine all methods and report ICs
    # ------------------------------------------------------------------

    def compare_combinations(
        self,
        signals: List[pd.Series],
        forward_returns: pd.Series,
        ic_window: int = 60,
    ) -> pd.DataFrame:
        """
        Run all four combination methods and report rolling IC statistics.

        Parameters
        ----------
        signals : list of pd.Series
        forward_returns : pd.Series
            Out-of-sample forward return series.
        ic_window : int

        Returns
        -------
        pd.DataFrame
            Summary with mean IC, ICIR, and t-stat for each method.
        """
        methods = {
            "equal_weight": self.equal_weight(signals),
            "ic_weighted": self.ic_weighted(
                signals,
                forward_returns=forward_returns,
                ic_window=ic_window,
            ),
            "pca_combine": self.pca_combine(signals, refit_window=ic_window),
            "hierarchical_combine": self.hierarchical_combine(signals),
        }

        rows = []
        for name, composite in methods.items():
            ic_series = _rolling_ic(composite, forward_returns, window=ic_window)
            ic_vals = ic_series.dropna()
            if len(ic_vals) == 0:
                rows.append(
                    {"method": name, "mean_ic": np.nan, "ic_std": np.nan,
                     "icir": np.nan, "t_stat": np.nan, "n_obs": 0}
                )
                continue
            mean_ic = ic_vals.mean()
            ic_std = ic_vals.std()
            icir = mean_ic / (ic_std + 1e-12)
            t_stat = mean_ic / (ic_std / np.sqrt(len(ic_vals)) + 1e-12)
            rows.append(
                {
                    "method": name,
                    "mean_ic": round(mean_ic, 4),
                    "ic_std": round(ic_std, 4),
                    "icir": round(icir, 4),
                    "t_stat": round(t_stat, 4),
                    "n_obs": len(ic_vals),
                }
            )

        return pd.DataFrame(rows).set_index("method")

    # ------------------------------------------------------------------
    # Dynamic weight allocation using recent IC
    # ------------------------------------------------------------------

    def dynamic_combine(
        self,
        signals: List[pd.Series],
        forward_returns: pd.Series,
        ic_window: int = 60,
        blend_methods: Optional[List[str]] = None,
        softmax_temp: float = 1.0,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Combine method-level composites with softmax weights derived from
        their rolling mean IC.

        Parameters
        ----------
        signals : list of pd.Series
        forward_returns : pd.Series
        ic_window : int
        blend_methods : list of str or None
            Subset of ['equal_weight','ic_weighted','pca_combine',
            'hierarchical_combine'].  Defaults to all four.
        softmax_temp : float
            Temperature for softmax weighting; lower = sharper.

        Returns
        -------
        composite : pd.Series
            Dynamically weighted composite signal.
        weights_df : pd.DataFrame
            Time-series of method weights.
        """
        if blend_methods is None:
            blend_methods = [
                "equal_weight",
                "ic_weighted",
                "pca_combine",
                "hierarchical_combine",
            ]

        method_composites: Dict[str, pd.Series] = {}
        for m in blend_methods:
            if m == "equal_weight":
                method_composites[m] = self.equal_weight(signals)
            elif m == "ic_weighted":
                method_composites[m] = self.ic_weighted(
                    signals,
                    forward_returns=forward_returns,
                    ic_window=ic_window,
                )
            elif m == "pca_combine":
                method_composites[m] = self.pca_combine(
                    signals, refit_window=ic_window
                )
            elif m == "hierarchical_combine":
                method_composites[m] = self.hierarchical_combine(signals)

        mc_df = pd.concat(list(method_composites.values()), axis=1)
        mc_df.columns = list(method_composites.keys())

        # Rolling IC for each method composite
        ic_series_dict: Dict[str, pd.Series] = {}
        for m, comp in method_composites.items():
            ic_series_dict[m] = _rolling_ic(comp, forward_returns, window=ic_window)

        ic_df = pd.concat(list(ic_series_dict.values()), axis=1).reindex(mc_df.index).ffill()
        ic_df.columns = list(method_composites.keys())

        rolling_mean_ic = ic_df.rolling(ic_window, min_periods=1).mean()

        # Softmax weights
        def _softmax(x: np.ndarray, temp: float) -> np.ndarray:
            x = x / (temp + 1e-12)
            x = x - x.max()
            e = np.exp(x)
            return e / (e.sum() + 1e-12)

        weights_list = []
        composite_vals = []

        for date in mc_df.index:
            ic_row = rolling_mean_ic.loc[date].values if date in rolling_mean_ic.index \
                else np.full(len(blend_methods), np.nan)
            comp_row = mc_df.loc[date].values

            if np.any(np.isnan(ic_row)) or np.any(np.isnan(comp_row)):
                composite_vals.append(np.nan)
                weights_list.append(np.full(len(blend_methods), np.nan))
                continue

            w = _softmax(ic_row, softmax_temp)
            weights_list.append(w)
            composite_vals.append(float(comp_row @ w))

        weights_df = pd.DataFrame(
            weights_list, index=mc_df.index, columns=blend_methods
        )
        composite = pd.Series(composite_vals, index=mc_df.index, name="dynamic_combine")

        return composite, weights_df

    # ------------------------------------------------------------------
    # Signal orthogonalization
    # ------------------------------------------------------------------

    def orthogonalize(
        self,
        signals: List[pd.Series],
        reference: Optional[pd.Series] = None,
    ) -> List[pd.Series]:
        """
        Orthogonalize signals against each other (or against a reference
        signal) via sequential OLS residualization.

        Parameters
        ----------
        signals : list of pd.Series
        reference : pd.Series, optional
            If given, each signal is orthogonalized against reference only.
            If None, Gram-Schmidt style sequential orthogonalization is used.

        Returns
        -------
        list of pd.Series
            Orthogonalized signals.
        """
        df = self._preprocess(signals)
        result = []

        if reference is not None:
            ref = reference.reindex(df.index).ffill()
            for col in df.columns:
                y = df[col]
                combined = pd.concat([y, ref], axis=1).dropna()
                if len(combined) < 20:
                    result.append(y.copy())
                    continue
                y_c = combined.iloc[:, 0].values
                x_c = combined.iloc[:, 1].values
                beta = np.dot(x_c, y_c) / (np.dot(x_c, x_c) + 1e-12)
                resid = y.copy()
                resid[combined.index] = y_c - beta * x_c
                resid.name = col
                result.append(resid)
        else:
            ortho_df = df.copy()
            for i in range(1, df.shape[1]):
                # Regress signal i on all previous orthogonalized signals
                col_i = df.columns[i]
                prev_cols = list(df.columns[:i])
                combined = ortho_df[prev_cols + [col_i]].dropna()
                if len(combined) < 20:
                    result_i = df[col_i].copy()
                else:
                    X = combined[prev_cols].values
                    y = combined[col_i].values
                    XtX = X.T @ X + 1e-8 * np.eye(X.shape[1])
                    beta = np.linalg.solve(XtX, X.T @ y)
                    resid_vals = y - X @ beta
                    result_i = df[col_i].copy()
                    result_i[combined.index] = resid_vals
                ortho_df[col_i] = result_i
                result.append(result_i if i > 0 else df.iloc[:, 0].copy())

            result = [df.iloc[:, 0].copy()] + result[1:]

        return result

    # ------------------------------------------------------------------
    # Turnover-adjusted combination
    # ------------------------------------------------------------------

    def turnover_adjusted(
        self,
        signals: List[pd.Series],
        max_daily_turnover: float = 0.10,
        decay: float = 0.95,
    ) -> pd.Series:
        """
        Equal-weight combination with turnover dampening.

        The output is an exponentially smoothed version of the raw composite
        that limits daily position changes to max_daily_turnover (fraction of
        notional) in expectation.  The effective half-life is derived from
        decay: hl = log(0.5)/log(decay).

        Parameters
        ----------
        signals : list of pd.Series
        max_daily_turnover : float
            Target maximum daily turnover fraction.
        decay : float
            EWM decay factor (0 < decay < 1).

        Returns
        -------
        pd.Series
            Turnover-adjusted composite signal.
        """
        raw = self.equal_weight(signals)
        # Exponential smoothing
        smoothed = raw.ewm(alpha=1 - decay, adjust=False).mean()
        smoothed.name = "turnover_adjusted"
        return smoothed

    # ------------------------------------------------------------------
    # Signal summary statistics
    # ------------------------------------------------------------------

    def signal_stats(
        self,
        signals: List[pd.Series],
        forward_returns: Optional[pd.Series] = None,
        ic_window: int = 60,
    ) -> pd.DataFrame:
        """
        Compute per-signal descriptive and predictive statistics.

        Parameters
        ----------
        signals : list of pd.Series
        forward_returns : pd.Series, optional
        ic_window : int

        Returns
        -------
        pd.DataFrame
            Per-signal statistics: mean, std, skew, kurt, autocorr, [mean_ic, icir].
        """
        df = _align_signals(signals)
        rows = []
        for col in df.columns:
            s = df[col].dropna()
            row: Dict = {
                "signal": col,
                "n_obs": len(s),
                "mean": round(s.mean(), 6),
                "std": round(s.std(), 6),
                "skew": round(float(s.skew()), 4),
                "kurt": round(float(s.kurt()), 4),
                "autocorr_1": round(float(s.autocorr(1)), 4) if len(s) > 1 else np.nan,
                "pct_nonzero": round((s != 0).mean(), 4),
            }
            if forward_returns is not None:
                ic_series = _rolling_ic(s, forward_returns, window=ic_window)
                ic_vals = ic_series.dropna()
                if len(ic_vals) > 0:
                    row["mean_ic"] = round(ic_vals.mean(), 4)
                    row["ic_std"] = round(ic_vals.std(), 4)
                    row["icir"] = round(ic_vals.mean() / (ic_vals.std() + 1e-12), 4)
                else:
                    row["mean_ic"] = np.nan
                    row["ic_std"] = np.nan
                    row["icir"] = np.nan
            rows.append(row)

        return pd.DataFrame(rows).set_index("signal")

    # ------------------------------------------------------------------
    # Pairwise correlation of signals
    # ------------------------------------------------------------------

    def pairwise_correlation(
        self, signals: List[pd.Series], method: str = "spearman"
    ) -> pd.DataFrame:
        """
        Compute pairwise rank or Pearson correlation between signals.

        Parameters
        ----------
        signals : list of pd.Series
        method : str
            'spearman' or 'pearson'.

        Returns
        -------
        pd.DataFrame
            Correlation matrix.
        """
        df = _align_signals(signals)
        if method == "spearman":
            return df.rank().corr()
        return df.corr()
