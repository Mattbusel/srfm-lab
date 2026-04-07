"""
research/portfolio_lab/correlation_labeler.py

Correlation structure analysis, regime labelling, and diversification
measures for SRFM-Lab.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------------
# Regime enum
# ---------------------------------------------------------------------------


class CorrelationRegime(Enum):
    """
    Correlation regime labels based on mean pairwise correlation.

    LOW:      mean pairwise < 0.30
    MODERATE: 0.30 <= mean pairwise < 0.60
    HIGH:     0.60 <= mean pairwise < 0.80
    CRISIS:   mean pairwise >= 0.80
    """

    LOW_CORR = "low"
    MODERATE_CORR = "moderate"
    HIGH_CORR = "high"
    CRISIS = "crisis"

    @classmethod
    def from_mean_corr(cls, mean_corr: float) -> "CorrelationRegime":
        if mean_corr < 0.30:
            return cls.LOW_CORR
        elif mean_corr < 0.60:
            return cls.MODERATE_CORR
        elif mean_corr < 0.80:
            return cls.HIGH_CORR
        else:
            return cls.CRISIS


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StabilityPoint:
    """Snapshot of correlation statistics at a given date."""

    date: pd.Timestamp
    mean_corr: float
    max_corr: float
    min_corr: float
    n_pairs_above_07: int
    n_pairs_total: int
    regime: CorrelationRegime

    @property
    def pct_pairs_above_07(self) -> float:
        if self.n_pairs_total == 0:
            return 0.0
        return self.n_pairs_above_07 / self.n_pairs_total


@dataclass
class ClusterResult:
    """Hierarchical clustering output."""

    cluster_labels: np.ndarray  # (n_assets,) integer cluster id
    n_clusters: int
    asset_names: List[str]
    # linkage matrix returned by scipy
    linkage_matrix: np.ndarray
    distance_threshold: float

    def cluster_members(self) -> dict:
        """Return dict mapping cluster_id -> list of asset names."""
        out: dict = {}
        for i, c in enumerate(self.cluster_labels):
            out.setdefault(int(c), []).append(self.asset_names[i])
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    """Extract upper-triangle (excluding diagonal) from a square matrix."""
    n = matrix.shape[0]
    idx = np.triu_indices(n, k=1)
    return matrix[idx]


def _mean_pairwise_corr(corr_matrix: pd.DataFrame) -> float:
    vals = _upper_triangle_values(corr_matrix.values)
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class CorrelationLabeler:
    """
    Analyse correlation structure of a multi-asset return DataFrame.

    Parameters
    ----------
    min_periods : minimum number of observations required to compute correlation
    """

    def __init__(self, min_periods: int = 20) -> None:
        self._min_periods = min_periods

    # ------------------------------------------------------------------
    # Build correlation matrix
    # ------------------------------------------------------------------

    def build_correlation_matrix(
        self,
        returns: pd.DataFrame,
        window: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Build the pairwise Pearson correlation matrix.

        Parameters
        ----------
        returns : daily returns with assets as columns
        window : if provided, use only the last `window` observations
        """
        data = returns.dropna()
        if window is not None:
            data = data.iloc[-window:]
        if len(data) < self._min_periods:
            raise ValueError(
                f"Only {len(data)} observations, need at least {self._min_periods}"
            )
        return data.corr()

    # ------------------------------------------------------------------
    # Regime labelling
    # ------------------------------------------------------------------

    def label_regime(self, corr_matrix: pd.DataFrame) -> CorrelationRegime:
        """
        Assign a CorrelationRegime based on mean pairwise correlation.
        """
        mean_c = _mean_pairwise_corr(corr_matrix)
        return CorrelationRegime.from_mean_corr(mean_c)

    # ------------------------------------------------------------------
    # Stability tracking
    # ------------------------------------------------------------------

    def track_correlation_stability(
        self,
        returns: pd.DataFrame,
        window: int = 63,
        step: int = 5,
    ) -> List[StabilityPoint]:
        """
        Roll a window over `returns` and compute correlation statistics
        at each step.

        Parameters
        ----------
        window : rolling window size in observations
        step   : number of observations between snapshots
        """
        data = returns.dropna()
        n = len(data)
        if n < window:
            raise ValueError(
                f"Returns length ({n}) shorter than window ({window})"
            )

        points: List[StabilityPoint] = []
        positions = range(window, n + 1, step)

        for end in positions:
            chunk = data.iloc[end - window : end]
            corr = chunk.corr().values
            vals = _upper_triangle_values(corr)
            if len(vals) == 0:
                continue
            mean_c = float(np.mean(vals))
            max_c = float(np.max(vals))
            min_c = float(np.min(vals))
            n_above = int(np.sum(vals > 0.7))
            date = data.index[end - 1]
            regime = CorrelationRegime.from_mean_corr(mean_c)
            points.append(
                StabilityPoint(
                    date=date,
                    mean_corr=mean_c,
                    max_corr=max_c,
                    min_corr=min_c,
                    n_pairs_above_07=n_above,
                    n_pairs_total=len(vals),
                    regime=regime,
                )
            )

        return points

    def stability_dataframe(self, points: List[StabilityPoint]) -> pd.DataFrame:
        """Convert stability point list to a tidy DataFrame."""
        rows = [
            {
                "date": p.date,
                "mean_corr": p.mean_corr,
                "max_corr": p.max_corr,
                "min_corr": p.min_corr,
                "n_pairs_above_07": p.n_pairs_above_07,
                "pct_pairs_above_07": p.pct_pairs_above_07,
                "regime": p.regime.value,
            }
            for p in points
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index("date")
        return df

    # ------------------------------------------------------------------
    # Cluster structure
    # ------------------------------------------------------------------

    def find_cluster_structure(
        self,
        corr_matrix: pd.DataFrame,
        distance_threshold: float = 0.5,
    ) -> ClusterResult:
        """
        Identify asset clusters using single-linkage hierarchical clustering.

        Distance is defined as sqrt(0.5 * (1 - corr)), a metric on correlations.
        Clusters are cut at `distance_threshold`.

        Parameters
        ----------
        corr_matrix : square Pearson correlation matrix
        distance_threshold : linkage distance to cut the dendrogram
        """
        asset_names = list(corr_matrix.columns)
        n = len(asset_names)

        corr_vals = corr_matrix.values.copy()
        # ensure symmetry and clip correlation to [-1, 1]
        corr_vals = np.clip((corr_vals + corr_vals.T) / 2.0, -1.0, 1.0)
        np.fill_diagonal(corr_vals, 1.0)

        # convert to distance matrix
        dist_vals = np.sqrt(0.5 * (1.0 - corr_vals))
        np.fill_diagonal(dist_vals, 0.0)

        # condensed distance for scipy
        condensed = squareform(dist_vals, checks=False)

        Z = linkage(condensed, method="single")
        labels = fcluster(Z, t=distance_threshold, criterion="distance")

        return ClusterResult(
            cluster_labels=labels,
            n_clusters=int(np.max(labels)),
            asset_names=asset_names,
            linkage_matrix=Z,
            distance_threshold=distance_threshold,
        )

    # ------------------------------------------------------------------
    # Rolling regime series
    # ------------------------------------------------------------------

    def rolling_regime_series(
        self,
        returns: pd.DataFrame,
        window: int = 63,
        step: int = 1,
    ) -> pd.Series:
        """
        Return a pd.Series (date index) of CorrelationRegime values.
        """
        points = self.track_correlation_stability(returns, window=window, step=step)
        idx = [p.date for p in points]
        vals = [p.regime for p in points]
        return pd.Series(vals, index=idx, name="correlation_regime")


# ---------------------------------------------------------------------------
# Diversification measures
# ---------------------------------------------------------------------------


class DiversificationMeasure:
    """
    Portfolio diversification and concentration metrics.
    """

    @staticmethod
    def effective_n(
        weights: np.ndarray,
        corr_matrix: Optional[np.ndarray] = None,
    ) -> float:
        """
        Effective number of assets.

        When corr_matrix is None (or all assets uncorrelated):
            effective_n = 1 / sum(w_i^2)   # Herfindahl-based
        When corr_matrix is provided:
            effective_n = 1 / (w^T * C * w)  where C is the correlation matrix
        This reduces to the weight-based version when C = I.
        """
        w = np.asarray(weights, dtype=float)
        if corr_matrix is None:
            denom = float(w @ w)
        else:
            C = np.asarray(corr_matrix, dtype=float)
            denom = float(w @ C @ w)
        if denom < 1e-15:
            return float(len(w))
        return 1.0 / denom

    @staticmethod
    def portfolio_concentration(weights: np.ndarray) -> float:
        """
        Herfindahl-Hirschman Index (HHI): sum of squared weights.

        Ranges from 1/n (fully diversified) to 1 (fully concentrated).
        """
        w = np.asarray(weights, dtype=float)
        return float(w @ w)

    @staticmethod
    def diversification_ratio(
        weights: np.ndarray,
        vols: np.ndarray,
        corr_matrix: np.ndarray,
    ) -> float:
        """
        Diversification ratio: weighted average volatility / portfolio volatility.

        DR = (w^T * sigma) / sqrt(w^T * Sigma * w)
        where Sigma = diag(sigma) * C * diag(sigma)

        A value of 1.0 means no diversification benefit.
        Values > 1 indicate diversification (components are not perfectly correlated).
        """
        w = np.asarray(weights, dtype=float)
        sigma = np.asarray(vols, dtype=float)
        C = np.asarray(corr_matrix, dtype=float)

        weighted_avg_vol = float(w @ sigma)

        # full covariance matrix
        Sigma = np.diag(sigma) @ C @ np.diag(sigma)
        port_var = float(w @ Sigma @ w)
        port_vol = float(np.sqrt(max(port_var, 0.0)))

        if port_vol < 1e-15:
            return 1.0

        return weighted_avg_vol / port_vol

    @staticmethod
    def risk_contributions(
        weights: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Compute each asset's fractional contribution to total portfolio risk.

        RC_i = w_i * (Sigma * w)_i / (w^T * Sigma * w)
        """
        w = np.asarray(weights, dtype=float)
        Sigma = np.asarray(cov_matrix, dtype=float)
        port_var = float(w @ Sigma @ w)
        if port_var < 1e-15:
            return np.ones(len(w)) / len(w)
        mrc = Sigma @ w
        rc = w * mrc / port_var
        return rc

    @staticmethod
    def gini_coefficient(weights: np.ndarray) -> float:
        """
        Gini coefficient of weights as a concentration measure.

        0 = perfect equality, 1 = fully concentrated in one asset.
        """
        w = np.sort(np.asarray(weights, dtype=float))
        n = len(w)
        if n == 0 or w.sum() < 1e-15:
            return 0.0
        cumw = np.cumsum(w)
        # Gini = 1 - 2 * area under Lorenz curve
        lorenz_area = float(np.sum(cumw[:-1])) / (n * float(cumw[-1]))
        return 1.0 - 2.0 * lorenz_area
