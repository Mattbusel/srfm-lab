"""
PCA-based factor extraction for quantitative research.

Implements:
- PCA factor extraction from return covariance matrix
- Explained variance analysis and scree plot data
- Varimax rotation for interpretable factors
- Factor mimicking portfolios (minimum-variance projection)
- Statistical factor model (Bai-Ng criterion for number of factors)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import svd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PCAResult:
    """Output of PCA factor extraction."""
    n_factors: int
    loadings: pd.DataFrame          # (tickers x factors) factor loading matrix
    factor_returns: pd.DataFrame    # (dates x factors) factor return series
    explained_variance: np.ndarray  # fraction of variance per factor
    cumulative_variance: np.ndarray
    eigenvalues: np.ndarray
    idiosyncratic_variance: pd.Series  # per-ticker residual variance
    rotation_matrix: Optional[np.ndarray]  # Varimax rotation if applied


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Demean and scale each column to unit variance."""
    mu = df.mean()
    sd = df.std().replace(0, 1.0)
    return (df - mu) / sd


def _correlation_matrix(df: pd.DataFrame) -> np.ndarray:
    """Pearson correlation matrix of columns."""
    return df.corr().values


def _covariance_matrix(df: pd.DataFrame) -> np.ndarray:
    return df.cov().values


def _varimax(loadings: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Raw Varimax rotation of factor loadings.

    Parameters
    ----------
    loadings : (p, k) array
        Unrotated factor loading matrix.
    max_iter : int
    tol : float

    Returns
    -------
    rotated_loadings : (p, k)
    rotation_matrix : (k, k)
    """
    p, k = loadings.shape
    R = np.eye(k)
    L = loadings.copy()

    for iteration in range(max_iter):
        old_R = R.copy()
        for i in range(k):
            for j in range(i + 1, k):
                # 2x2 rotation in (i, j) plane
                L_ij = L[:, [i, j]]
                u = L_ij[:, 0] ** 2 - L_ij[:, 1] ** 2
                v = 2 * L_ij[:, 0] * L_ij[:, 1]
                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u ** 2 - v ** 2)
                D = 2 * np.sum(u * v)
                num = D - 2 * A * B / p
                den = C - (A ** 2 - B ** 2) / p
                angle = 0.25 * np.arctan2(num, den)
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                L[:, [i, j]] = L_ij @ rot
                R[:, [i, j]] = R[:, [i, j]] @ rot

        if np.max(np.abs(R - old_R)) < tol:
            break

    return L, R


# ---------------------------------------------------------------------------
# Main PCA class
# ---------------------------------------------------------------------------

class StatisticalFactorModel:
    """
    Statistical (PCA-based) factor model for equity returns.

    Extracts latent factors from the cross-section of asset returns via
    PCA on the sample covariance (or correlation) matrix.

    Parameters
    ----------
    n_factors : int or 'auto'
        Number of factors to extract.  'auto' uses Bai-Ng ICp2 criterion.
    use_correlation : bool
        If True, run PCA on correlation matrix (standardized returns).
        If False, use covariance matrix.
    max_factors_auto : int
        Maximum number of factors when n_factors='auto'.
    apply_varimax : bool
        Apply Varimax rotation to improve interpretability.
    min_obs_ratio : float
        Minimum T/N ratio required for reliable PCA.
    """

    def __init__(
        self,
        n_factors: int | str = 5,
        use_correlation: bool = True,
        max_factors_auto: int = 15,
        apply_varimax: bool = False,
        min_obs_ratio: float = 2.0,
    ) -> None:
        self.n_factors = n_factors
        self.use_correlation = use_correlation
        self.max_factors_auto = max_factors_auto
        self.apply_varimax = apply_varimax
        self.min_obs_ratio = min_obs_ratio

    def _bai_ng_criterion(
        self,
        returns: pd.DataFrame,
        max_k: int,
    ) -> int:
        """
        Bai-Ng (2002) ICp2 information criterion for number of factors.

        IC(k) = log(V(k)) + k * (N + T) / (N * T) * log(N * T / (N + T))

        where V(k) = sum of squared residuals / (N*T).

        Returns
        -------
        int
            Optimal number of factors.
        """
        T, N = returns.shape
        X = returns.values.copy()
        X = np.nan_to_num(X, nan=0.0)

        ic_values = []
        for k in range(1, max_k + 1):
            # SVD
            U, s, Vt = svd(X, full_matrices=False)
            F = U[:, :k] * s[:k]  # (T, k) factor matrix
            L = Vt[:k, :].T       # (N, k) loading matrix
            X_hat = F @ L.T
            residuals = X - X_hat
            V_k = np.sum(residuals ** 2) / (N * T)
            penalty = k * (N + T) / (N * T) * np.log(N * T / (N + T))
            ic_values.append(np.log(V_k) + penalty)

        return int(np.argmin(ic_values) + 1)

    def fit(self, returns: pd.DataFrame, demean: bool = True) -> PCAResult:
        """
        Fit the statistical factor model.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset return matrix (dates x tickers).
        demean : bool
            If True, demean each column before PCA.

        Returns
        -------
        PCAResult
        """
        clean = returns.dropna(how="all", axis=1).dropna(how="any", axis=0)
        T, N = clean.shape
        assert T / N >= self.min_obs_ratio or N <= 50, (
            f"Too few observations (T={T}) for N={N} assets; "
            f"require T/N >= {self.min_obs_ratio}"
        )

        if demean:
            X = clean.values - clean.mean().values
        else:
            X = clean.values.copy()

        if self.use_correlation:
            sd = clean.std().values
            sd[sd == 0] = 1.0
            X_scaled = X / sd
        else:
            X_scaled = X
            sd = np.ones(N)

        # Determine number of factors
        if self.n_factors == "auto":
            k = self._bai_ng_criterion(pd.DataFrame(X_scaled), self.max_factors_auto)
        else:
            k = min(int(self.n_factors), min(T, N) - 1)

        # SVD of X_scaled: X = U S V^T
        U, s, Vt = svd(X_scaled, full_matrices=False)
        # Factor returns: (T, k) = U * S
        F = U[:, :k] * s[:k]
        # Loadings: (N, k) = V
        L = Vt[:k, :].T  # (N, k)

        # Normalize: F has unit variance per factor, rescale loadings
        factor_std = F.std(axis=0, ddof=1)
        factor_std[factor_std == 0] = 1.0
        F = F / factor_std
        L = L * factor_std

        # If we standardized by sd, convert loadings back to return space
        if self.use_correlation:
            L = L * sd[:, None]

        # Explained variance
        eigenvalues = s[:k] ** 2 / (T - 1)
        total_variance = np.sum(X_scaled ** 2) / (T - 1) / N
        all_eigenvalues = s ** 2 / (T - 1)
        explained = eigenvalues / (np.sum(all_eigenvalues) + 1e-12)
        cumulative = np.cumsum(explained)

        # Varimax rotation
        rotation_matrix = None
        if self.apply_varimax and k > 1:
            L_rotated, R = _varimax(L)
            F_rotated = F @ R
            L = L_rotated
            F = F_rotated
            rotation_matrix = R

        # Sign convention: largest absolute loading positive per factor
        for j in range(k):
            if np.abs(L[:, j]).argmax() != np.argmax(L[:, j]):
                L[:, j] = -L[:, j]
                F[:, j] = -F[:, j]

        # Idiosyncratic variance
        X_hat = F @ L.T
        resid = X - X_hat  # (T, N)
        idio_var = np.var(resid, axis=0, ddof=1)

        factor_names = [f"PC{i+1}" for i in range(k)]
        loading_df = pd.DataFrame(L, index=clean.columns, columns=factor_names)
        factor_df = pd.DataFrame(F, index=clean.index, columns=factor_names)
        idio_series = pd.Series(idio_var, index=clean.columns, name="idio_var")

        return PCAResult(
            n_factors=k,
            loadings=loading_df,
            factor_returns=factor_df,
            explained_variance=explained,
            cumulative_variance=cumulative,
            eigenvalues=eigenvalues,
            idiosyncratic_variance=idio_series,
            rotation_matrix=rotation_matrix,
        )

    def scree_data(self, returns: pd.DataFrame, max_factors: int = 20) -> pd.DataFrame:
        """
        Compute explained variance for 1..max_factors.

        Returns
        -------
        pd.DataFrame
            Columns: n_factors, explained_variance, cumulative_variance.
        """
        clean = returns.dropna(how="all", axis=1).dropna(how="any", axis=0)
        X = clean.values - clean.mean().values
        if self.use_correlation:
            sd = clean.std().values
            sd[sd == 0] = 1.0
            X = X / sd
        T, N = X.shape
        max_factors = min(max_factors, min(T, N) - 1)

        _, s, _ = svd(X, full_matrices=False)
        eigenvalues = s ** 2 / (T - 1)
        total = eigenvalues.sum()

        rows = []
        cumulative = 0.0
        for i in range(max_factors):
            ev = eigenvalues[i] / total
            cumulative += ev
            rows.append({
                "n_factors": i + 1,
                "eigenvalue": round(float(eigenvalues[i]), 6),
                "explained_variance": round(float(ev), 6),
                "cumulative_variance": round(cumulative, 6),
            })
        return pd.DataFrame(rows).set_index("n_factors")

    def factor_mimicking_portfolios(
        self,
        result: PCAResult,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Construct factor mimicking portfolios (minimum-variance projection).

        The mimicking portfolio for factor j has weights w_j such that:
          w_j = (Sigma^{-1} L_j) / (L_j' Sigma^{-1} L_j)

        where Sigma is the covariance matrix of idiosyncratic returns.

        Parameters
        ----------
        result : PCAResult
        returns : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            (tickers x factors) portfolio weights.
        """
        clean = returns.dropna(how="all", axis=1).dropna(how="any", axis=0)
        common_assets = result.loadings.index.intersection(clean.columns)
        L = result.loadings.loc[common_assets].values
        N, k = L.shape

        # Estimate idiosyncratic covariance (diagonal)
        idio_var = result.idiosyncratic_variance.reindex(common_assets).values
        idio_var = np.maximum(idio_var, 1e-8)
        Sigma_inv = np.diag(1.0 / idio_var)  # simplified diagonal approx

        weights_matrix = np.zeros((N, k))
        for j in range(k):
            l_j = L[:, j]
            SigInv_l = Sigma_inv @ l_j
            denom = l_j @ SigInv_l
            if abs(denom) > 1e-10:
                weights_matrix[:, j] = SigInv_l / denom

        factor_names = [f"PC{i+1}" for i in range(k)]
        weights_df = pd.DataFrame(
            weights_matrix, index=common_assets, columns=factor_names
        )
        return weights_df

    def factor_model_residuals(
        self,
        result: PCAResult,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute residuals from the fitted factor model.

        Parameters
        ----------
        result : PCAResult
        returns : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Idiosyncratic return residuals.
        """
        common_dates = result.factor_returns.index.intersection(returns.index)
        common_assets = result.loadings.index.intersection(returns.columns)

        F = result.factor_returns.loc[common_dates].values  # (T, k)
        L = result.loadings.loc[common_assets].values        # (N, k)
        R = returns.loc[common_dates, common_assets].values  # (T, N)

        # Handle NaN in returns
        R_clean = np.nan_to_num(R, nan=0.0)
        fitted = F @ L.T  # (T, N)
        resid = R_clean - fitted

        # Restore NaN mask
        nan_mask = np.isnan(R)
        resid[nan_mask] = np.nan

        return pd.DataFrame(resid, index=common_dates, columns=common_assets)

    def explained_variance_by_asset(
        self,
        result: PCAResult,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """
        Per-asset R² explained by the factor model.

        Returns
        -------
        pd.Series
            R² per ticker.
        """
        resid_df = self.factor_model_residuals(result, returns)
        common_assets = resid_df.columns.intersection(returns.columns)

        r2_vals = {}
        for asset in common_assets:
            r = resid_df[asset].dropna()
            y = returns[asset].reindex(r.index).dropna()
            if len(y) < 10:
                r2_vals[asset] = np.nan
                continue
            ss_res = np.sum(r.loc[y.index] ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2_vals[asset] = round(1 - ss_res / (ss_tot + 1e-12), 4)

        return pd.Series(r2_vals, name="r_squared")

    def factor_correlation(self, result: PCAResult) -> pd.DataFrame:
        """Pairwise correlation between factor return series."""
        return result.factor_returns.corr().round(4)

    def factor_summary(self, result: PCAResult) -> pd.DataFrame:
        """Summary statistics for each extracted factor."""
        rows = []
        for col in result.factor_returns.columns:
            f = result.factor_returns[col]
            i = int(col.replace("PC", "")) - 1
            rows.append({
                "factor": col,
                "eigenvalue": round(float(result.eigenvalues[i]), 6),
                "explained_variance": round(float(result.explained_variance[i]), 6),
                "cumulative_variance": round(float(result.cumulative_variance[i]), 6),
                "mean_annual": round(float(f.mean() * 252), 4),
                "vol_annual": round(float(f.std() * np.sqrt(252)), 4),
                "sharpe": round(float(f.mean() / (f.std() + 1e-12) * np.sqrt(252)), 4),
                "skew": round(float(f.skew()), 4),
                "max_abs_loading": round(float(result.loadings[col].abs().max()), 4),
            })
        return pd.DataFrame(rows).set_index("factor")

    def top_loadings(
        self,
        result: PCAResult,
        n_top: int = 10,
        factor: str = "PC1",
    ) -> pd.Series:
        """
        Return the top n assets by absolute loading on a given factor.

        Parameters
        ----------
        result : PCAResult
        n_top : int
        factor : str

        Returns
        -------
        pd.Series
            Top loadings sorted by absolute value.
        """
        if factor not in result.loadings.columns:
            return pd.Series(dtype=float)
        loadings = result.loadings[factor].dropna()
        return loadings.reindex(loadings.abs().nlargest(n_top).index)

    def risk_decomposition(
        self,
        result: PCAResult,
        portfolio_weights: pd.Series,
    ) -> Dict[str, float]:
        """
        Decompose portfolio variance into systematic and idiosyncratic components.

        Parameters
        ----------
        result : PCAResult
        portfolio_weights : pd.Series
            Asset weights (need not sum to 1).

        Returns
        -------
        dict
            systematic_variance, idiosyncratic_variance, total_variance,
            per-factor variances.
        """
        common = result.loadings.index.intersection(portfolio_weights.index)
        w = portfolio_weights.reindex(common).fillna(0).values
        L = result.loadings.loc[common].values  # (N, k)

        # Factor exposures of portfolio: (k,)
        beta_port = L.T @ w

        # Factor variances (use eigenvalues as proxy for factor variance)
        k = len(beta_port)
        factor_vars = result.eigenvalues[:k]

        # Systematic variance
        systematic_var = float(np.sum(beta_port ** 2 * factor_vars))

        # Idiosyncratic variance
        idio_var = result.idiosyncratic_variance.reindex(common).fillna(0).values
        idiosyncratic_var = float(np.sum(w ** 2 * idio_var))

        total_var = systematic_var + idiosyncratic_var

        output = {
            "systematic_variance": round(systematic_var, 8),
            "idiosyncratic_variance": round(idiosyncratic_var, 8),
            "total_variance": round(total_var, 8),
            "systematic_pct": round(systematic_var / (total_var + 1e-12), 4),
        }
        factor_names = [f"PC{i+1}" for i in range(k)]
        for i, fname in enumerate(factor_names):
            output[f"var_{fname}"] = round(float(beta_port[i] ** 2 * factor_vars[i]), 8)

        return output
