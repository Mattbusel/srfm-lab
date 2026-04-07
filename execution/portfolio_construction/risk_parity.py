# execution/portfolio_construction/risk_parity.py
# Risk parity portfolio construction with multiple covariance estimators.
#
# Implements:
#   - Equal risk contribution (ERC) via gradient descent / SLSQP
#   - Risk budgeting (arbitrary target budgets)
#   - Volatility parity (1/vol naive weighting)
#   - Risk contribution decomposition
#   - CovarianceEstimator: sample, Ledoit-Wolf, EWMA, RMT denoising

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RiskContribution:
    """Per-asset risk decomposition for a given portfolio."""

    asset: str
    weight: float
    marginal_risk_contribution: float  # d(sigma_p) / d(w_i)
    percent_risk_contribution: float   # fraction of total portfolio risk


# ---------------------------------------------------------------------------
# RiskParityOptimizer
# ---------------------------------------------------------------------------


class RiskParityOptimizer:
    """
    Constructs risk-parity and risk-budgeted portfolios.

    All methods accept a covariance matrix as a 2-D numpy array with shape
    (n, n) and return a weight vector of shape (n,).

    The covariance matrix is assumed to be expressed in consistent time units
    (e.g., annualised daily covariance).  Weights are long-only and sum to 1
    unless otherwise stated.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def equal_risk_contribution(
        self,
        cov_matrix: np.ndarray,
        bounds: Tuple[float, float] = (0.01, 0.30),
    ) -> np.ndarray:
        """
        Compute equal risk contribution (ERC) weights.

        Solves:
            min  sum_i sum_j (RC_i - RC_j)^2
            s.t. sum(w) = 1
                 bounds[0] <= w_i <= bounds[1]  for all i

        where RC_i = w_i * (Sigma * w)_i / sqrt(w^T Sigma w)

        Uses SLSQP via scipy.optimize.minimize.  An equal-weight portfolio is
        used as the starting point and as a fallback if optimisation fails.

        Parameters
        ----------
        cov_matrix : np.ndarray, shape (n, n)
            Positive semi-definite covariance matrix.
        bounds : tuple (lo, hi)
            Per-asset weight bounds.  Defaults to (0.01, 0.30).

        Returns
        -------
        np.ndarray, shape (n,)
            ERC weights summing to 1.
        """
        cov_matrix = _ensure_psd(cov_matrix)
        n = cov_matrix.shape[0]
        w0 = np.full(n, 1.0 / n)

        def _objective(w: np.ndarray) -> float:
            rc = _risk_contributions_raw(w, cov_matrix)
            # Pairwise squared differences -- O(n^2) but n <= 50 in practice.
            total = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    total += (rc[i] - rc[j]) ** 2
            return total

        def _gradient(w: np.ndarray) -> np.ndarray:
            # Finite-difference gradient; analytic form is messy and error-prone.
            eps = 1e-7
            grad = np.zeros(n)
            f0 = _objective(w)
            for k in range(n):
                w_up = w.copy()
                w_up[k] += eps
                grad[k] = (_objective(w_up) - f0) / eps
            return grad

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bound_list = [bounds] * n

        result = minimize(
            _objective,
            w0,
            jac=_gradient,
            method="SLSQP",
            bounds=bound_list,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 2000, "disp": False},
        )

        if not result.success:
            warnings.warn(
                f"ERC optimisation did not converge: {result.message}. "
                "Returning best iterate found.",
                RuntimeWarning,
                stacklevel=2,
            )

        weights = np.clip(result.x, bounds[0], bounds[1])
        weights /= weights.sum()
        return weights

    def risk_budgeted(
        self,
        cov_matrix: np.ndarray,
        budgets: np.ndarray,
        bounds: Tuple[float, float] = (0.01, 0.30),
    ) -> np.ndarray:
        """
        Compute risk-budgeted weights given target risk budgets.

        Solves:
            min  sum_i (RC_i - b_i)^2
            s.t. sum(w) = 1
                 bounds[0] <= w_i <= bounds[1]

        where b_i are the target risk fraction for asset i (must sum to 1).

        Parameters
        ----------
        cov_matrix : np.ndarray, shape (n, n)
        budgets : np.ndarray, shape (n,)
            Target fractional risk contributions.  Will be normalised to sum
            to 1 internally.
        bounds : tuple (lo, hi)

        Returns
        -------
        np.ndarray, shape (n,)
        """
        cov_matrix = _ensure_psd(cov_matrix)
        n = cov_matrix.shape[0]
        budgets = np.asarray(budgets, dtype=float)
        if budgets.shape[0] != n:
            raise ValueError(
                f"budgets length {budgets.shape[0]} != cov_matrix size {n}"
            )
        budgets = budgets / budgets.sum()

        w0 = budgets.copy()
        w0 = np.clip(w0, bounds[0], bounds[1])
        w0 /= w0.sum()

        def _objective(w: np.ndarray) -> float:
            rc = _risk_contributions_raw(w, cov_matrix)
            return float(np.sum((rc - budgets) ** 2))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bound_list = [bounds] * n

        result = minimize(
            _objective,
            w0,
            method="SLSQP",
            bounds=bound_list,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 2000, "disp": False},
        )

        if not result.success:
            warnings.warn(
                f"Risk budget optimisation did not converge: {result.message}.",
                RuntimeWarning,
                stacklevel=2,
            )

        weights = np.clip(result.x, bounds[0], bounds[1])
        weights /= weights.sum()
        return weights

    def vol_parity(self, vols: np.ndarray) -> np.ndarray:
        """
        Compute 1/vol (volatility parity) weights.

        Ignores correlations; simply allocates inversely proportional to
        each asset's volatility.

        Parameters
        ----------
        vols : np.ndarray, shape (n,)
            Per-asset volatilities (must be positive).

        Returns
        -------
        np.ndarray, shape (n,)
        """
        vols = np.asarray(vols, dtype=float)
        if np.any(vols <= 0):
            raise ValueError("All volatilities must be strictly positive.")
        inv_vol = 1.0 / vols
        return inv_vol / inv_vol.sum()

    def compute_risk_contributions(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        asset_names: Optional[Sequence[str]] = None,
    ) -> List[RiskContribution]:
        """
        Decompose portfolio risk into per-asset contributions.

        RC_i = w_i * (Sigma * w)_i / sqrt(w^T * Sigma * w)

        Parameters
        ----------
        weights : np.ndarray, shape (n,)
        cov_matrix : np.ndarray, shape (n, n)
        asset_names : sequence of str, optional
            If not provided, assets are labelled 'asset_0', 'asset_1', ...

        Returns
        -------
        list[RiskContribution]
        """
        weights = np.asarray(weights, dtype=float)
        cov_matrix = np.asarray(cov_matrix, dtype=float)
        n = weights.shape[0]
        if asset_names is None:
            asset_names = [f"asset_{i}" for i in range(n)]

        rc = _risk_contributions_raw(weights, cov_matrix)
        portfolio_vol = float(np.sqrt(weights @ cov_matrix @ weights))
        marginal_rc = (cov_matrix @ weights) / portfolio_vol  # d(sigma_p)/d(w_i)
        pct_rc = rc  # already fractional contributions

        result = []
        for i in range(n):
            result.append(
                RiskContribution(
                    asset=str(asset_names[i]),
                    weight=float(weights[i]),
                    marginal_risk_contribution=float(marginal_rc[i]),
                    percent_risk_contribution=float(pct_rc[i]),
                )
            )
        return result

    def get_effective_n(self, weights: np.ndarray) -> float:
        """
        Effective N diversification metric.

        Defined as 1 / sum(w_i^2) -- the Herfindahl-based measure.
        Ranges from 1 (fully concentrated) to n (equally weighted).

        Parameters
        ----------
        weights : np.ndarray, shape (n,)

        Returns
        -------
        float
        """
        weights = np.asarray(weights, dtype=float)
        sq_sum = float(np.sum(weights ** 2))
        if sq_sum == 0.0:
            return 0.0
        return 1.0 / sq_sum


# ---------------------------------------------------------------------------
# CovarianceEstimator
# ---------------------------------------------------------------------------


class CovarianceEstimator:
    """
    Multiple covariance estimation methods for use in portfolio construction.

    All methods accept a pd.DataFrame of returns (rows = dates, cols = assets)
    and return a 2-D numpy array of shape (n_assets, n_assets).
    """

    # ------------------------------------------------------------------
    # Sample covariance
    # ------------------------------------------------------------------

    def sample_cov(
        self,
        returns: pd.DataFrame,
        min_periods: int = 60,
    ) -> np.ndarray:
        """
        Standard sample covariance matrix.

        Parameters
        ----------
        returns : pd.DataFrame
            Return series; rows are observations, columns are assets.
        min_periods : int
            Minimum number of non-NaN observations required.  Assets with
            fewer observations are dropped and the result is re-indexed to
            the full set (with NaN-filled rows replaced by zeros).

        Returns
        -------
        np.ndarray, shape (n_assets, n_assets)
        """
        if len(returns) < min_periods:
            raise ValueError(
                f"Need at least {min_periods} observations, got {len(returns)}."
            )
        cov = returns.cov(min_periods=min_periods).values.astype(float)
        # Replace any NaNs from insufficient overlap with 0 (conservative).
        cov = np.nan_to_num(cov, nan=0.0)
        return _ensure_psd(cov)

    # ------------------------------------------------------------------
    # Ledoit-Wolf analytical shrinkage
    # ------------------------------------------------------------------

    def ledoit_wolf(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Analytical Ledoit-Wolf shrinkage toward the scaled-identity target.

        Implements the Oracle Approximating Shrinkage (OAS-style) formula from
        Ledoit & Wolf (2004) "A Well-Conditioned Estimator for Large-Dimensional
        Covariance Matrices".

        Shrinkage: Sigma_LW = (1 - rho*) * S + rho* * mu * I
        where rho* is the optimal shrinkage intensity and mu = trace(S)/n.

        Parameters
        ----------
        returns : pd.DataFrame

        Returns
        -------
        np.ndarray, shape (n, n)
        """
        X = returns.values.astype(float)
        # Remove NaN rows (pairwise would require more complex logic).
        X = X[~np.isnan(X).any(axis=1)]
        T, n = X.shape

        if T <= 1:
            raise ValueError("Need at least 2 observations for Ledoit-Wolf.")

        # Demean.
        X = X - X.mean(axis=0)

        S = (X.T @ X) / T  # sample cov (biased for analytic formula)

        # Analytical shrinkage intensity rho* (LW 2004, Theorem 1).
        # Target: F = mu * I  where mu = trace(S) / n
        mu = np.trace(S) / n

        # Frobenius norm squared of S minus mu*I
        S_minus_mu = S - mu * np.eye(n)
        delta = np.sum(S_minus_mu ** 2) / n  # ||S - mu*I||_F^2 / n

        # Asymptotic variance of the shrinkage estimator.
        # sum_t of (||x_t x_t^T - S||_F^2) / T^2
        sum_sq = 0.0
        for t in range(T):
            x = X[t, :]
            outer = np.outer(x, x)
            diff = outer - S
            sum_sq += np.sum(diff ** 2)
        beta_bar = sum_sq / (T * T)

        # Optimal rho*
        beta = min(beta_bar, delta)
        rho_star = beta / delta if delta > 0 else 0.0
        rho_star = np.clip(rho_star, 0.0, 1.0)

        cov_lw = (1.0 - rho_star) * S + rho_star * mu * np.eye(n)
        return _ensure_psd(cov_lw)

    # ------------------------------------------------------------------
    # EWMA covariance (RiskMetrics)
    # ------------------------------------------------------------------

    def ewma_cov(
        self,
        returns: pd.DataFrame,
        lambda_: float = 0.94,
    ) -> np.ndarray:
        """
        Exponentially weighted moving average covariance (RiskMetrics 1994).

        Sigma_t = lambda * Sigma_{t-1} + (1 - lambda) * r_{t-1} r_{t-1}^T

        Returns the covariance estimate as of the last observation.

        Parameters
        ----------
        returns : pd.DataFrame
        lambda_ : float
            Decay factor.  RiskMetrics uses 0.94 for daily and 0.97 for
            monthly data.

        Returns
        -------
        np.ndarray, shape (n, n)
        """
        X = returns.values.astype(float)
        X = np.nan_to_num(X, nan=0.0)
        T, n = X.shape

        if T < 2:
            raise ValueError("Need at least 2 observations for EWMA covariance.")

        # Initialise with sample covariance of first 20 observations (or all
        # if T < 20).
        init_window = min(20, T)
        cov = np.cov(X[:init_window].T, bias=True)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])

        for t in range(init_window, T):
            r = X[t, :].reshape(-1, 1)
            cov = lambda_ * cov + (1.0 - lambda_) * (r @ r.T)

        return _ensure_psd(cov)

    # ------------------------------------------------------------------
    # RMT / Marchenko-Pastur denoised covariance
    # ------------------------------------------------------------------

    def denoised_cov(
        self,
        returns: pd.DataFrame,
        clip_pct: float = 0.90,
    ) -> np.ndarray:
        """
        Random Matrix Theory denoised covariance (Marchenko-Pastur clipping).

        Algorithm:
        1. Compute correlation matrix C from sample covariance.
        2. Eigendecompose C.
        3. Determine the Marchenko-Pastur upper bound:
               lambda_+ = sigma^2 * (1 + sqrt(q))^2
           where q = n / T and sigma^2 is estimated as the mean eigenvalue
           of the "bulk" (assumed noise) part.
        4. Clip eigenvalues below the upper bound to their mean, preserving
           trace (total variance).
        5. Reconstruct the correlation matrix and re-scale to covariance.

        Parameters
        ----------
        returns : pd.DataFrame
        clip_pct : float
            Fraction of eigenvalues considered as noise (sorted ascending).
            If 0.90, the bottom 90% are subject to MP clipping.

        Returns
        -------
        np.ndarray, shape (n, n)
        """
        X = returns.values.astype(float)
        X = X[~np.isnan(X).any(axis=1)]
        T, n = X.shape

        if T < n:
            warnings.warn(
                "More assets than observations -- RMT denoising may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Sample covariance and correlation.
        X_dm = X - X.mean(axis=0)
        S = (X_dm.T @ X_dm) / (T - 1)
        stds = np.sqrt(np.diag(S))
        stds[stds == 0] = 1.0
        D_inv = np.diag(1.0 / stds)
        C = D_inv @ S @ D_inv  # correlation matrix

        # Eigendecomposition of correlation matrix.
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        # eigh returns in ascending order.
        eigenvalues = np.clip(eigenvalues, 0.0, None)  # numerical safety

        # Marchenko-Pastur upper bound.
        q = n / T
        # sigma^2 estimate: mean of the eigenvalues in the noise bulk.
        n_noise = max(1, int(clip_pct * n))
        sigma2 = float(np.mean(eigenvalues[:n_noise]))
        lambda_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2

        # Replace eigenvalues below lambda_plus with their mean (preserve trace).
        noise_mask = eigenvalues < lambda_plus
        if noise_mask.any():
            noise_mean = float(eigenvalues[noise_mask].mean())
            # We replace noise eigenvalues to preserve total variance.
            # Standard approach: set to the mean noise eigenvalue but scale so
            # that the sum of noise eigenvalues is preserved.
            n_noise_actual = int(noise_mask.sum())
            noise_total = float(eigenvalues[noise_mask].sum())
            fill_value = noise_total / n_noise_actual if n_noise_actual > 0 else noise_mean
            eigenvalues[noise_mask] = fill_value

        # Reconstruct correlation matrix.
        C_denoised = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Force diagonal to 1 (numerical precision).
        diag_vals = np.diag(C_denoised)
        diag_vals = np.where(diag_vals <= 0, 1.0, diag_vals)
        D_corr = np.diag(1.0 / np.sqrt(diag_vals))
        C_denoised = D_corr @ C_denoised @ D_corr

        # Scale back to covariance.
        D_std = np.diag(stds)
        cov_denoised = D_std @ C_denoised @ D_std

        return _ensure_psd(cov_denoised)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _risk_contributions_raw(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Return fractional risk contributions (sum to 1).

    RC_i = w_i * (Sigma * w)_i / (w^T * Sigma * w)^{0.5}
    Then normalise so sum(RC) = 1.
    """
    w = np.asarray(w, dtype=float)
    cov = np.asarray(cov, dtype=float)
    Sw = cov @ w
    port_var = float(w @ Sw)
    if port_var <= 0:
        return np.full(len(w), 1.0 / len(w))
    port_vol = np.sqrt(port_var)
    # Absolute risk contribution (units of portfolio vol).
    arc = w * Sw / port_vol
    # Normalise to fractions.
    arc_sum = arc.sum()
    if arc_sum <= 0:
        return np.full(len(w), 1.0 / len(w))
    return arc / arc_sum


def _ensure_psd(cov: np.ndarray, min_eigenvalue: float = 1e-8) -> np.ndarray:
    """
    Project a covariance matrix onto the positive semi-definite cone.

    Clips any eigenvalues below min_eigenvalue to min_eigenvalue, then
    reconstructs the matrix.  Returns a symmetric PSD matrix.
    """
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2.0  # enforce symmetry first
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, min_eigenvalue, None)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


# ---------------------------------------------------------------------------
# Module-level convenience instances
# ---------------------------------------------------------------------------

_default_rp_optimizer = RiskParityOptimizer()
_default_cov_estimator = CovarianceEstimator()
