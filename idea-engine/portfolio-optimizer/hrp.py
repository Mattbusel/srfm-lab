"""
portfolio-optimizer/hrp.py

Hierarchical Risk Parity (HRP) implementation following López de Prado (2016).

HRP allocates portfolio weights by:
  1. Computing a correlation-based distance matrix.
  2. Clustering assets via Ward linkage.
  3. Quasi-diagonalising the covariance matrix according to the cluster order.
  4. Recursively bisecting clusters and allocating inverse-variance weights
     within each cluster.

This approach avoids inverting the covariance matrix, making it more robust
than traditional mean-variance optimisation when the number of assets is large
or the matrix is near-singular.

Reference:
  López de Prado, M. (2016). Building diversified portfolios that outperform
  out-of-sample. Journal of Portfolio Management, 42(4), 59-69.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HRPResult:
    """
    Result from a Hierarchical Risk Parity optimisation.

    Attributes
    ----------
    weights : pd.Series
        Optimal HRP weights indexed by asset name.
    sorted_assets : list[str]
        Asset names in the quasi-diagonalised cluster order.
    linkage_matrix : np.ndarray
        Ward linkage matrix (n-1 × 4) for visualisation / inspection.
    cluster_variance : dict[str, float]
        Cluster-level variance for each bisection step.
    """

    weights: pd.Series
    sorted_assets: list[str]
    linkage_matrix: np.ndarray
    cluster_variance: dict[str, float]

    def as_dict(self) -> dict[str, float]:
        """Return weights as a plain dict {asset: weight}."""
        return self.weights.to_dict()

    def as_array(self, asset_order: list[str]) -> np.ndarray:
        """Return weights as a numpy array in the requested asset order."""
        return self.weights.reindex(asset_order).fillna(0.0).values


# ---------------------------------------------------------------------------
# Core HRP implementation
# ---------------------------------------------------------------------------


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity portfolio construction.

    Parameters
    ----------
    linkage_method : str
        Linkage algorithm for hierarchical clustering.  'ward' (default)
        is recommended; 'single', 'complete', 'average' also supported.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> returns = pd.DataFrame(rng.normal(0, 0.01, (500, 6)),
    ...                        columns=list("ABCDEF"))
    >>> hrp = HierarchicalRiskParity()
    >>> result = hrp.fit(returns)
    >>> print(result.weights.round(4))
    """

    def __init__(self, linkage_method: str = "ward") -> None:
        self.linkage_method = linkage_method

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, returns: pd.DataFrame) -> HRPResult:
        """
        Run the full HRP algorithm on a returns DataFrame.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns (each column = one asset).

        Returns
        -------
        HRPResult
            Optimal HRP weights and intermediate diagnostics.
        """
        clean = returns.dropna()
        assets = list(clean.columns)
        cov = clean.cov().values
        corr = clean.corr().values

        # Step 1: distance matrix
        dist_matrix = self.correlation_distance(corr)

        # Step 2: hierarchical clustering
        link = linkage(squareform(dist_matrix), method=self.linkage_method)

        # Step 3: quasi-diagonalisation
        sorted_items = self.quasi_diagonalize(link, list(range(len(assets))))

        # Step 4: recursive bisection
        weights_arr, cv = self.recursive_bisection(cov, sorted_items)

        weights = pd.Series(
            weights_arr,
            index=[assets[i] for i in sorted_items],
            name="hrp_weight",
        )
        # Re-index to original asset order
        weights = weights.reindex(assets)

        return HRPResult(
            weights=weights,
            sorted_assets=[assets[i] for i in sorted_items],
            linkage_matrix=link,
            cluster_variance=cv,
        )

    def fit_from_cov(
        self,
        cov_matrix: np.ndarray,
        asset_names: Optional[list[str]] = None,
    ) -> HRPResult:
        """
        Run HRP directly from a covariance matrix (no returns required).

        Parameters
        ----------
        cov_matrix : np.ndarray
            Square covariance matrix (n × n).
        asset_names : list[str], optional
            Asset names; defaults to ['asset_0', 'asset_1', ...].

        Returns
        -------
        HRPResult
        """
        n = cov_matrix.shape[0]
        assets = asset_names or [f"asset_{i}" for i in range(n)]

        # Derive correlation from covariance
        vols = np.sqrt(np.diag(cov_matrix))
        vols = np.where(vols == 0, 1, vols)
        corr = cov_matrix / np.outer(vols, vols)
        np.fill_diagonal(corr, 1.0)

        dist_matrix = self.correlation_distance(corr)
        link = linkage(squareform(dist_matrix), method=self.linkage_method)
        sorted_items = self.quasi_diagonalize(link, list(range(n)))
        weights_arr, cv = self.recursive_bisection(cov_matrix, sorted_items)

        weights = pd.Series(
            weights_arr,
            index=[assets[i] for i in sorted_items],
            name="hrp_weight",
        ).reindex(assets)

        return HRPResult(
            weights=weights,
            sorted_assets=[assets[i] for i in sorted_items],
            linkage_matrix=link,
            cluster_variance=cv,
        )

    # ------------------------------------------------------------------
    # Step 1: Distance matrix
    # ------------------------------------------------------------------

    def correlation_distance(self, corr_matrix: np.ndarray) -> np.ndarray:
        """
        Compute a correlation-based distance matrix.

        Distance metric: d(i,j) = sqrt(0.5 * (1 - ρᵢⱼ))

        This satisfies the triangle inequality and maps ρ=-1 → d=1,
        ρ=0 → d=0.707, ρ=1 → d=0.

        Parameters
        ----------
        corr_matrix : np.ndarray
            Square correlation matrix (n × n).

        Returns
        -------
        np.ndarray
            Symmetric distance matrix with zeros on the diagonal.
        """
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr_matrix), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)
        return dist

    # ------------------------------------------------------------------
    # Step 2: Quasi-diagonalisation
    # ------------------------------------------------------------------

    def quasi_diagonalize(
        self,
        link: np.ndarray,
        items: list[int],
    ) -> list[int]:
        """
        Reorder the items (leaf nodes) according to the hierarchical clustering
        linkage, so that similar assets appear adjacent in the matrix
        (quasi-diagonalisation).

        Parameters
        ----------
        link : np.ndarray
            Linkage matrix from :func:`scipy.cluster.hierarchy.linkage`.
        items : list[int]
            Initial list of asset indices (leaves), typically ``list(range(n))``.

        Returns
        -------
        list[int]
            Reordered asset indices.
        """
        link = link.astype(int)
        n = len(items)

        # Build a mapping: cluster_id → list of leaf indices
        cluster_items: dict[int, list[int]] = {i: [items[i]] for i in range(n)}

        for merge_idx, row in enumerate(link):
            left, right = int(row[0]), int(row[1])
            new_id = n + merge_idx
            cluster_items[new_id] = cluster_items[left] + cluster_items[right]

        # The last merge contains all items in the quasi-diagonalised order
        root_id = n + len(link) - 1
        return cluster_items[root_id]

    # ------------------------------------------------------------------
    # Step 3: Recursive bisection
    # ------------------------------------------------------------------

    def recursive_bisection(
        self,
        cov_matrix: np.ndarray,
        sorted_items: list[int],
    ) -> tuple[np.ndarray, dict[str, float]]:
        """
        Allocate weights via top-down recursive bisection of the sorted
        cluster tree, assigning inverse-variance weights within each cluster.

        The weight of each sub-cluster is proportional to the inverse of that
        cluster's variance, ensuring risk is split equally between the two
        sub-trees at each bisection level.

        Parameters
        ----------
        cov_matrix : np.ndarray
            Full asset covariance matrix (n × n).
        sorted_items : list[int]
            Quasi-diagonalised asset indices (output of :meth:`quasi_diagonalize`).

        Returns
        -------
        (weights_array, cluster_variance_log)
            - weights_array : np.ndarray of shape (n,) in the order of sorted_items
            - cluster_variance_log : dict mapping cluster description to variance
        """
        n = len(sorted_items)
        weights = np.ones(n)  # start with equal weights
        cluster_variance_log: dict[str, float] = {}

        # Stack contains (start, end) index pairs into sorted_items
        subsets: list[list[int]] = [list(range(n))]

        while subsets:
            subset = subsets.pop()
            if len(subset) <= 1:
                continue

            # Bisect
            mid = len(subset) // 2
            left_idx = subset[:mid]
            right_idx = subset[mid:]

            # Variance of each sub-cluster
            var_left = self.cluster_variance(cov_matrix, [sorted_items[i] for i in left_idx])
            var_right = self.cluster_variance(cov_matrix, [sorted_items[i] for i in right_idx])

            key = f"split_{sorted_items[left_idx[0]]}-{sorted_items[right_idx[-1]]}"
            cluster_variance_log[key] = float(var_left + var_right)

            # Alpha: fraction allocated to left sub-cluster
            total_var = var_left + var_right
            if total_var == 0:
                alpha = 0.5
            else:
                alpha = float(1.0 - var_left / total_var)

            # Scale current weights
            weights[left_idx] *= alpha
            weights[right_idx] *= (1.0 - alpha)

            subsets.append(left_idx)
            subsets.append(right_idx)

        # Normalise to sum to 1
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights /= weight_sum

        return weights, cluster_variance_log

    def cluster_variance(
        self,
        cov_matrix: np.ndarray,
        items: list[int],
    ) -> float:
        """
        Compute the variance of an inverse-variance-weighted sub-portfolio
        composed of the given asset indices.

        Parameters
        ----------
        cov_matrix : np.ndarray
            Full covariance matrix.
        items : list[int]
            Asset indices forming this sub-cluster.

        Returns
        -------
        float
            Sub-portfolio variance under inverse-variance weighting.
        """
        if len(items) == 1:
            return float(cov_matrix[items[0], items[0]])

        sub_cov = cov_matrix[np.ix_(items, items)]
        diag_inv_var = 1.0 / np.diag(sub_cov)
        diag_inv_var_sum = diag_inv_var.sum()
        if diag_inv_var_sum == 0:
            return 0.0
        w = diag_inv_var / diag_inv_var_sum  # inverse-variance weights
        return float(w @ sub_cov @ w)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def plot_dendrogram(
        self,
        link: np.ndarray,
        labels: Optional[list[str]] = None,
    ) -> None:
        """
        Display a dendrogram for the HRP clustering (requires matplotlib).

        Parameters
        ----------
        link : np.ndarray
            Linkage matrix.
        labels : list[str], optional
            Asset labels for the leaf nodes.
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 4))
            dendrogram(link, labels=labels, ax=ax)
            ax.set_title("HRP Cluster Dendrogram")
            ax.set_ylabel("Ward Distance")
            plt.tight_layout()
            plt.show()
        except ImportError:
            raise RuntimeError("matplotlib is required for plot_dendrogram")
