"""
research/portfolio_lab/construction.py

Portfolio construction methods for SRFM-Lab.

All portfolio classes expose:
    fit(returns)                    -> dict[str, float]  (weights)
    optimize(returns, constraints)  -> dict[str, float]  (constrained weights)

Returns input: pd.DataFrame of shape (T, N) where columns are asset names
               and rows are periodic returns (e.g. daily).

Implementations:
    EqualWeightPortfolio       — 1/N allocation
    InverseVolPortfolio        — weight = 1/σ_i, normalised
    MinVariancePortfolio       — global minimum variance via scipy
    MaxSharpePortfolio         — tangency portfolio via scipy
    HRPPortfolio               — Hierarchical Risk Parity (López de Prado 2016)
    BlackLittermanPortfolio    — CAPM equilibrium + investor views blend
    RiskParityPortfolio        — Equal risk contribution
    KellyPortfolio             — Multi-asset Kelly with correlation adjustment
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BasePortfolio(ABC):
    """Abstract base for portfolio construction methods."""

    def fit(self, returns: pd.DataFrame) -> dict[str, float]:
        """
        Compute optimal portfolio weights.

        Args:
            returns : pd.DataFrame of shape (T, N), columns = asset names,
                      values = periodic returns (e.g. daily log or simple).

        Returns:
            Dict mapping asset name to weight. Weights sum to 1.0.
        """
        self._validate(returns)
        weights = self._fit(returns)
        return self._normalise(weights)

    def optimize(
        self,
        returns: pd.DataFrame,
        constraints: Optional[dict] = None,
    ) -> dict[str, float]:
        """
        Compute constrained portfolio weights.

        Args:
            returns     : Returns DataFrame.
            constraints : Dict with optional keys:
                - 'min_weight'  : float, minimum per-asset weight (default 0.0)
                - 'max_weight'  : float, maximum per-asset weight (default 1.0)
                - 'long_only'   : bool, force non-negative weights (default True)
                - 'target_vol'  : float, target annualised volatility.

        Returns:
            Dict of constrained weights.
        """
        self._validate(returns)
        c = constraints or {}
        weights = self._fit_constrained(returns, c)
        return self._normalise(weights)

    @abstractmethod
    def _fit(self, returns: pd.DataFrame) -> dict[str, float]:
        ...

    def _fit_constrained(
        self, returns: pd.DataFrame, constraints: dict
    ) -> dict[str, float]:
        """Default: apply bounds post-hoc to unconstrained solution."""
        weights = self._fit(returns)
        assets = list(weights.keys())
        w = np.array([weights[a] for a in assets])

        long_only = constraints.get("long_only", True)
        min_w = constraints.get("min_weight", 0.0 if long_only else -1.0)
        max_w = constraints.get("max_weight", 1.0)

        w = np.clip(w, min_w, max_w)
        total = w.sum()
        if abs(total) < 1e-12:
            w = np.ones(len(assets)) / len(assets)
        else:
            w /= total
        return dict(zip(assets, w.tolist()))

    @staticmethod
    def _validate(returns: pd.DataFrame) -> None:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pd.DataFrame.")
        if returns.empty:
            raise ValueError("returns DataFrame is empty.")
        if returns.shape[1] < 1:
            raise ValueError("returns must have at least 1 column.")

    @staticmethod
    def _normalise(weights: dict[str, float]) -> dict[str, float]:
        total = sum(weights.values())
        if abs(total) < 1e-12:
            n = len(weights)
            return {k: 1.0 / n for k in weights}
        return {k: v / total for k, v in weights.items()}

    @staticmethod
    def _cov_matrix(returns: pd.DataFrame, min_periods: int = 20) -> np.ndarray:
        cov = returns.cov()
        return cov.values.astype(np.float64)

    @staticmethod
    def _mean_returns(returns: pd.DataFrame) -> np.ndarray:
        return returns.mean().values.astype(np.float64)


# ---------------------------------------------------------------------------
# EqualWeightPortfolio
# ---------------------------------------------------------------------------


class EqualWeightPortfolio(BasePortfolio):
    """
    1/N equal weight portfolio.

    Ignores all statistical properties of returns.
    Robust benchmark for comparison.
    """

    def _fit(self, returns: pd.DataFrame) -> dict[str, float]:
        n = returns.shape[1]
        return {col: 1.0 / n for col in returns.columns}


# ---------------------------------------------------------------------------
# InverseVolPortfolio
# ---------------------------------------------------------------------------


class InverseVolPortfolio(BasePortfolio):
    """
    Inverse-volatility portfolio: weight_i = (1/σ_i) / Σ(1/σ_j).

    Allocates more to lower-volatility assets.
    Ignores correlations between assets.

    Args:
        vol_window : Rolling window for volatility estimation. None = full sample.
        ann_factor : Annualisation factor (252 for daily).
    """

    def __init__(self, vol_window: Optional[int] = None, ann_factor: int = 252) -> None:
        self.vol_window = vol_window
        self.ann_factor = ann_factor

    def _fit(self, returns: pd.DataFrame) -> dict[str, float]:
        if self.vol_window is not None:
            vols = returns.tail(self.vol_window).std() * math.sqrt(self.ann_factor)
        else:
            vols = returns.std() * math.sqrt(self.ann_factor)

        inv_vol = 1.0 / (vols + 1e-12)
        weights = (inv_vol / inv_vol.sum()).to_dict()
        return {str(k): float(v) for k, v in weights.items()}


# ---------------------------------------------------------------------------
# MinVariancePortfolio
# ---------------------------------------------------------------------------


class MinVariancePortfolio(BasePortfolio):
    """
    Global minimum variance portfolio.

    Minimises w^T Σ w subject to Σw = 1 and w >= 0.

    Args:
        long_only  : Enforce non-negative weights.
        max_weight : Maximum single-asset weight.
        cov_method : Covariance estimator: 'sample' | 'ledoit_wolf' | 'shrinkage'.
    """

    def __init__(
        self,
        long_only: bool = True,
        max_weight: float = 1.0,
        cov_method: str = "ledoit_wolf",
    ) -> None:
        self.long_only = long_only
        self.max_weight = max_weight
        self.cov_method = cov_method

    def _fit(self, returns: pd.DataFrame) -> dict[str, float]:
        assets = list(returns.columns)
        n = len(assets)
        Sigma = _estimate_covariance(returns, self.cov_method)

        w0 = np.ones(n) / n

        def portfolio_variance(w):
            return float(w @ Sigma @ w)

        def grad_variance(w):
            return 2.0 * Sigma @ w

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds = [(0.0 if self.long_only else -1.0, self.max_weight)] * n

        result = minimize(
            portfolio_variance,
            w0,
            jac=grad_variance,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 500},
        )

        if not result.success:
            warnings.warn(f"MinVariance optimisation did not converge: {result.message}")

        w = np.clip(result.x, 0.0 if self.long_only else -1.0, self.max_weight)
        return dict(zip(assets, w.tolist()))


# ---------------------------------------------------------------------------
# MaxSharpePortfolio
# ---------------------------------------------------------------------------


class MaxSharpePortfolio(BasePortfolio):
    """
    Maximum Sharpe ratio (tangency) portfolio.

    Maximises (w^T μ - rf) / sqrt(w^T Σ w) via the Markowitz efficient frontier.

    Args:
        risk_free_rate : Annual risk-free rate.
        long_only      : Enforce non-negative weights.
        max_weight     : Maximum single-asset weight.
        cov_method     : Covariance estimation method.
        ann_factor     : Annualisation factor.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.04,
        long_only: bool = True,
        max_weight: float = 1.0,
        cov_method: str = "ledoit_wolf",
        ann_factor: int = 252,
    ) -> None:
        self.rf = risk_free_rate
        self.long_only = long_only
        self.max_weight = max_weight
        self.cov_method = cov_method
        self.ann_factor = ann_factor

    def _fit(self, returns: pd.DataFrame) -> dict[str, float]:
        assets = list(returns.columns)
        n = len(assets)
        mu = self._mean_returns(returns) * self.ann_factor
        Sigma = _estimate_covariance(returns, self.cov_method) * self.ann_factor
        rf_daily = self.rf / self.ann_factor

        def neg_sharpe(w):
            port_return = float(w @ mu)
            port_vol = math.sqrt(max(float(w @ Sigma @ w), 1e-12))
            return -(port_return - self.rf) / port_vol

        def grad_neg_sharpe(w):
            ret = float(w @ mu)
            var = float(w @ Sigma @ w)
            vol = math.sqrt(max(var, 1e-12))
            sharpe = (ret - self.rf) / vol
            d_ret = mu
            d_vol = (Sigma @ w) / vol
            return -(d_ret * vol - (ret - self.rf) * d_vol) / (vol ** 2)

        w0 = np.ones(n) / n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds = [(0.0 if self.long_only else -1.0, self.max_weight)] * n

        result = minimize(
            neg_sharpe,
            w0,
            jac=grad_neg_sharpe,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )

        if not result.success:
            warnings.warn(f"MaxSharpe optimisation did not converge: {result.message}")

        w = np.clip(result.x, 0.0 if self.long_only else -1.0, self.max_weight)
        return dict(zip(assets, w.tolist()))

    def efficient_frontier(
        self, returns: pd.DataFrame, n_points: int = 50
    ) -> pd.DataFrame:
        """
        Compute the mean-variance efficient frontier.

        Returns:
            DataFrame with columns: ['return', 'volatility', 'sharpe', 'weights'].
        """
        assets = list(returns.columns)
        n = len(assets)
        mu = self._mean_returns(returns) * self.ann_factor
        Sigma = _estimate_covariance(returns, self.cov_method) * self.ann_factor

        min_ret = float(np.min(mu))
        max_ret = float(np.max(mu))
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier = []
        for target in target_returns:
            constraints = [
                {"type": "eq", "fun": lambda w: w.sum() - 1.0},
                {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t},
            ]
            bounds = [(0.0, 1.0)] * n
            result = minimize(
                lambda w: float(w @ Sigma @ w),
                np.ones(n) / n,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-12},
            )
            if result.success:
                w = result.x
                vol = math.sqrt(max(float(w @ Sigma @ w), 1e-12))
                sharpe = (target - self.rf) / vol
                frontier.append({
                    "return": target,
                    "volatility": vol,
                    "sharpe": sharpe,
                    "weights": dict(zip(assets, w.tolist())),
                })

        return pd.DataFrame(frontier)


# ---------------------------------------------------------------------------
# HRPPortfolio — Hierarchical Risk Parity
# ---------------------------------------------------------------------------


class HRPPortfolio(BasePortfolio):
    """
    Hierarchical Risk Parity portfolio (López de Prado, 2016).

    Algorithm:
        1. Compute correlation matrix from returns.
        2. Build correlation-based distance matrix.
        3. Hierarchical clustering (Ward linkage).
        4. Quasi-diagonalise the covariance matrix.
        5. Recursive bisection to allocate weights.

    This approach avoids inverting the covariance matrix, making it
    more robust than mean-variance optimisation.

    Args:
        linkage_method : Scipy linkage method ('single', 'complete', 'ward', 'average').
        cov_method     : Covariance estimation method.
    """

    def __init__(
        self,
        linkage_method: str = "ward",
        cov_method: str = "ledoit_wolf",
    ) -> None:
        self.linkage_method = linkage_method
        self.cov_method = cov_method

    def _fit(self, returns: pd.DataFrame) -> dict[str, float]:
        assets = list(returns.columns)
        n = len(assets)

        if n == 1:
            return {assets[0]: 1.0}

        Sigma = _estimate_covariance(returns, self.cov_method)
        corr = _cov_to_corr(Sigma)

        # Distance matrix: d_ij = sqrt(0.5*(1 - rho_ij))
        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)

        # Hierarchical clustering
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method=self.linkage_method)

        # Get sorted leaf order from dendrogram
        dend = dendrogram(link, no_plot=True)
        sorted_indices = dend["leaves"]

        # Recursive bisection
        weights = self._recursive_bisect(Sigma, sorted_indices)

        return {assets[i]: float(weights[i]) for i in range(n)}

    def _recursive_bisect(
        self, Sigma: np.ndarray, sorted_items: list[int]
    ) -> np.ndarray:
        """
        Allocate weights via recursive bisection on the sorted cluster order.

        At each split, allocate capital to the two sub-clusters inversely
        proportional to their variance (cluster variance = w^T Σ w using
        inverse-vol weights within the cluster).
        """
        n = len(Sigma)
        weights = np.ones(n)

        def _cluster_var(items: list[int]) -> float:
            sub_sigma = Sigma[np.ix_(items, items)]
            vols = np.sqrt(np.maximum(np.diag(sub_sigma), 1e-12))
            inv_vol_w = 1.0 / vols
            inv_vol_w /= inv_vol_w.sum()
            return float(inv_vol_w @ sub_sigma @ inv_vol_w)

        def _bisect(items: list[int]) -> None:
            if len(items) <= 1:
                return

            split = len(items) // 2
            left = items[:split]
            right = items[split:]

            var_left = _cluster_var(left)
            var_right = _cluster_var(right)

            total_var = var_left + var_right + 1e-12
            alpha = 1.0 - var_left / total_var

            # Scale down left cluster by (1-alpha) and right by alpha
            for idx in left:
                weights[idx] *= 1.0 - alpha
            for idx in right:
                weights[idx] *= alpha

            _bisect(left)
            _bisect(right)

        _bisect(sorted_items)
        return weights


# ---------------------------------------------------------------------------
# BlackLittermanPortfolio
# ---------------------------------------------------------------------------


class BlackLittermanPortfolio(BasePortfolio):
    """
    Black-Litterman portfolio construction.

    Blends CAPM equilibrium returns with investor views.

    Args:
        risk_aversion : Market risk-aversion coefficient λ (default 2.5).
        tau           : Prior scaling factor (default 0.05).
        cov_method    : Covariance estimation method.
        ann_factor    : Annualisation factor.
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        cov_method: str = "ledoit_wolf",
        ann_factor: int = 252,
    ) -> None:
        self.delta = risk_aversion
        self.tau = tau
        self.cov_method = cov_method
        self.ann_factor = ann_factor
        self.views: Optional[dict] = None

    def set_views(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: Optional[np.ndarray] = None,
    ) -> "BlackLittermanPortfolio":
        """
        Set investor views for the Black-Litterman model.

        Args:
            P     : View matrix of shape (K, N) where K = number of views,
                    N = number of assets. Each row specifies a relative or
                    absolute view on assets.
            Q     : View returns vector of shape (K,). Expected returns for each view.
            Omega : View uncertainty covariance (K, K). If None, uses diagonal
                    with proportional-to-P*Sigma*P^T scaling.

        Returns:
            self (for method chaining).
        """
        self.views = {"P": np.asarray(P, dtype=np.float64), "Q": np.asarray(Q, dtype=np.float64), "Omega": Omega}
        return self

    def _fit(self, returns: pd.DataFrame) -> dict[str, float]:
        assets = list(returns.columns)
        n = len(assets)
        Sigma = _estimate_covariance(returns, self.cov_method) * self.ann_factor

        # Market weights: use equal weights as market cap proxy
        w_mkt = np.ones(n) / n

        # Equilibrium (implied) excess returns: π = δ * Σ * w_mkt
        pi = self.delta * Sigma @ w_mkt

        if self.views is None:
            # Without views: use equilibrium returns with MinVar optimisation
            mu_bl = pi
        else:
            P = self.views["P"]
            Q = self.views["Q"]
            Omega = self.views["Omega"]

            if Omega is None:
                # Diagonal uncertainty proportional to variance of views
                Omega = np.diag(np.diag(self.tau * P @ Sigma @ P.T))

            # Black-Litterman posterior:
            # M1 = (tau*Σ)^-1
            # M2 = P^T * Omega^-1 * P
            # mu_bl = [(tau*Σ)^-1 + P^T * Omega^-1 * P]^-1 * [(tau*Σ)^-1 * pi + P^T * Omega^-1 * Q]
            tau_sigma = self.tau * Sigma
            try:
                tau_sigma_inv = np.linalg.inv(tau_sigma + np.eye(n) * 1e-8)
                Omega_inv = np.linalg.inv(Omega + np.eye(len(Omega)) * 1e-8)
            except np.linalg.LinAlgError:
                tau_sigma_inv = np.linalg.pinv(tau_sigma)
                Omega_inv = np.linalg.pinv(Omega)

            M = tau_sigma_inv + P.T @ Omega_inv @ P
            try:
                M_inv = np.linalg.inv(M + np.eye(n) * 1e-8)
            except np.linalg.LinAlgError:
                M_inv = np.linalg.pinv(M)

            mu_bl = M_inv @ (tau_sigma_inv @ pi + P.T @ Omega_inv @ Q)

        # Optimal weights: w = (1/δ) * Σ^-1 * μ_bl
        try:
            sigma_inv = np.linalg.inv(Sigma + np.eye(n) * 1e-8)
        except np.linalg.LinAlgError:
            sigma_inv = np.linalg.pinv(Sigma)

        w = (1.0 / self.delta) * sigma_inv @ mu_bl

        # Long-only constraint with normalisation
        w = np.maximum(w, 0.0)
        total = w.sum()
        if total < 1e-12:
            w = np.ones(n) / n
        else:
            w /= total

        return dict(zip(assets, w.tolist()))


# ---------------------------------------------------------------------------
# RiskParityPortfolio — Equal Risk Contribution
# ---------------------------------------------------------------------------


class RiskParityPortfolio(BasePortfolio):
    """
    Risk Parity: Equal Risk Contribution (ERC) portfolio.

    Finds weights such that each asset contributes equally to total
    portfolio variance: RC_i = w_i * (Σw)_i = σ_p^2 / N for all i.

    Args:
        cov_method : Covariance estimation method.
        max_iter   : Maximum solver iterations.
        tol        : Convergence tolerance.
    """

    def __init__(
        self,
        cov_method: str = "ledoit_wolf",
        max_iter: int = 1000,
        tol: float = 1e-10,
    ) -> None:
        self.cov_method = cov_method
        self.max_iter = max_iter
        self.tol = tol

    def _fit(self, returns: pd.DataFrame) -> dict[str, float]:
        assets = list(returns.columns)
        n = len(assets)
        Sigma = _estimate_covariance(returns, self.cov_method)

        w = np.ones(n) / n
        target_rc = 1.0 / n  # equal risk contribution fraction

        for iteration in range(self.max_iter):
            port_var = float(w @ Sigma @ w)
            if port_var < 1e-12:
                break
            marginal_contrib = Sigma @ w
            risk_contrib = w * marginal_contrib / port_var

            # Gradient: minimise sum(RC_i - 1/n)^2
            grad = 2.0 * (risk_contrib - target_rc) * (
                marginal_contrib / port_var
                - w * (port_var + 2.0 * w @ Sigma @ w) / (port_var ** 2)
            )
            # Clamp gradient step
            step = 0.01 / (np.linalg.norm(grad) + 1e-12)
            w_new = w - step * grad
            w_new = np.maximum(w_new, 1e-8)
            w_new /= w_new.sum()

            change = float(np.linalg.norm(w_new - w))
            w = w_new
            if change < self.tol:
                break

        return dict(zip(assets, w.tolist()))


# ---------------------------------------------------------------------------
# KellyPortfolio — Multi-asset Kelly
# ---------------------------------------------------------------------------


class KellyPortfolio(BasePortfolio):
    """
    Multi-asset Kelly criterion with correlation adjustment.

    Full Kelly: f* = Σ^{-1} μ  (unconstrained)
    Fractional Kelly: scale by fraction parameter.

    The optimal Kelly fraction maximises log-expected wealth over a single period.

    Args:
        fraction   : Kelly fraction (1.0 = full Kelly, 0.5 = half Kelly).
        ann_factor : Annualisation factor.
        long_only  : Enforce non-negative weights.
        max_weight : Maximum per-asset weight.
        cov_method : Covariance estimation method.
    """

    def __init__(
        self,
        fraction: float = 0.5,
        ann_factor: int = 252,
        long_only: bool = True,
        max_weight: float = 0.5,
        cov_method: str = "ledoit_wolf",
    ) -> None:
        self.fraction = fraction
        self.ann_factor = ann_factor
        self.long_only = long_only
        self.max_weight = max_weight
        self.cov_method = cov_method

    def _fit(self, returns: pd.DataFrame) -> dict[str, float]:
        assets = list(returns.columns)
        n = len(assets)
        mu = self._mean_returns(returns) * self.ann_factor
        Sigma = _estimate_covariance(returns, self.cov_method) * self.ann_factor

        # Full Kelly: f* = Σ^{-1} μ
        try:
            Sigma_inv = np.linalg.inv(Sigma + np.eye(n) * 1e-8)
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.pinv(Sigma)

        kelly_weights = Sigma_inv @ mu

        # Apply fractional Kelly
        kelly_weights *= self.fraction

        # Apply constraints
        if self.long_only:
            kelly_weights = np.maximum(kelly_weights, 0.0)

        kelly_weights = np.clip(kelly_weights, 0.0 if self.long_only else -self.max_weight, self.max_weight)

        total = kelly_weights.sum()
        if abs(total) < 1e-12:
            kelly_weights = np.ones(n) / n
        else:
            kelly_weights /= total

        return dict(zip(assets, kelly_weights.tolist()))

    def growth_rate(self, weights: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
        """
        Expected log growth rate: g = w^T μ - 0.5 * w^T Σ w.

        Args:
            weights : Portfolio weights, shape (N,).
            mu      : Expected returns, shape (N,).
            Sigma   : Covariance matrix, shape (N, N).

        Returns:
            Scalar growth rate.
        """
        return float(weights @ mu - 0.5 * weights @ Sigma @ weights)


# ---------------------------------------------------------------------------
# Covariance estimation utilities
# ---------------------------------------------------------------------------


def _estimate_covariance(returns: pd.DataFrame, method: str = "ledoit_wolf") -> np.ndarray:
    """
    Estimate the covariance matrix of returns.

    Args:
        returns : Returns DataFrame (T, N).
        method  : 'sample' | 'ledoit_wolf' | 'shrinkage' | 'oas' | 'minimum_covariance_determinant'.

    Returns:
        Estimated covariance matrix of shape (N, N).
    """
    X = returns.values.astype(np.float64)
    n, p = X.shape

    if method == "sample":
        return np.cov(X.T)

    elif method == "ledoit_wolf":
        return _ledoit_wolf(X)

    elif method == "shrinkage":
        # Analytical Ledoit-Wolf shrinkage (Oracle Approximating)
        return _ledoit_wolf(X)

    elif method == "oas":
        return _oas(X)

    elif method == "minimum_covariance_determinant":
        return _robust_mcd_cov(X)

    else:
        raise ValueError(f"Unknown covariance method: {method}. "
                         f"Choose: sample, ledoit_wolf, shrinkage, oas, minimum_covariance_determinant")


def _ledoit_wolf(X: np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf analytical shrinkage estimator.

    Shrinks sample covariance toward scaled identity matrix:
        Σ_lw = (1-α) * S + α * μ * I

    where α is the optimal shrinkage intensity and μ = tr(S)/p.
    """
    n, p = X.shape
    X_centered = X - X.mean(axis=0)
    S = (X_centered.T @ X_centered) / n  # sample covariance (biased)

    # Oracle approximating shrinkage (Ledoit & Wolf 2004 analytical formula)
    mu_target = np.trace(S) / p

    # Frobenius norm squared of S
    S_sq = float(np.sum(S ** 2))
    # Variance of sample covariance elements
    delta_sq = 0.0
    for i in range(n):
        xi = X_centered[i]
        zi = np.outer(xi, xi)
        delta_sq += float(np.sum((zi - S) ** 2))
    delta_sq /= n ** 2

    gamma = float(np.sum((S - mu_target * np.eye(p)) ** 2))

    if gamma < 1e-12:
        return S

    alpha = min(1.0, delta_sq / gamma)

    return (1.0 - alpha) * S + alpha * mu_target * np.eye(p)


def _oas(X: np.ndarray) -> np.ndarray:
    """
    Oracle Approximating Shrinkage (OAS) estimator (Chen et al., 2010).

    Shrinks the sample covariance toward the scaled identity.
    """
    n, p = X.shape
    X_centered = X - X.mean(axis=0)
    S = (X_centered.T @ X_centered) / n

    trace_S = np.trace(S)
    trace_S2 = np.trace(S @ S)

    # OAS shrinkage coefficient
    rho_num = (1.0 - 2.0 / p) * trace_S2 + trace_S ** 2
    rho_den = (n + 1.0 - 2.0 / p) * (trace_S2 - trace_S ** 2 / p)

    rho = min(1.0, max(0.0, rho_num / (rho_den + 1e-12)))
    mu_target = trace_S / p

    return (1.0 - rho) * S + rho * mu_target * np.eye(p)


def _robust_mcd_cov(X: np.ndarray, support_fraction: float = 0.75) -> np.ndarray:
    """
    Minimum Covariance Determinant estimator (approximate).

    Uses the C-step algorithm: subsample, estimate, refine.
    Returns a robust covariance matrix less sensitive to outliers.
    """
    n, p = X.shape
    h = max(p + 1, int(support_fraction * n))

    rng = np.random.default_rng(42)
    best_det = np.inf
    best_cov = np.cov(X.T)

    # Try multiple random starts
    n_starts = min(10, n // h + 1)
    for _ in range(n_starts):
        idx = rng.choice(n, size=h, replace=False)
        subset = X[idx]
        mu = subset.mean(axis=0)
        cov = np.cov(subset.T) + np.eye(p) * 1e-8

        # C-step: refine by keeping h points with smallest Mahalanobis distance
        for _ in range(10):  # iterate
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov)
            diff = X - mu
            maha = np.array([float(d @ cov_inv @ d) for d in diff])
            idx = np.argsort(maha)[:h]
            subset = X[idx]
            mu = subset.mean(axis=0)
            cov = np.cov(subset.T) + np.eye(p) * 1e-8

        det = float(np.linalg.det(cov))
        if det < best_det:
            best_det = det
            best_cov = cov

    return best_cov


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to correlation matrix."""
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_METHOD_MAP: dict[str, type] = {
    "equal": EqualWeightPortfolio,
    "equal_weight": EqualWeightPortfolio,
    "ew": EqualWeightPortfolio,
    "inverse_vol": InverseVolPortfolio,
    "inv_vol": InverseVolPortfolio,
    "min_variance": MinVariancePortfolio,
    "min_var": MinVariancePortfolio,
    "gmv": MinVariancePortfolio,
    "max_sharpe": MaxSharpePortfolio,
    "tangency": MaxSharpePortfolio,
    "hrp": HRPPortfolio,
    "hierarchical": HRPPortfolio,
    "black_litterman": BlackLittermanPortfolio,
    "bl": BlackLittermanPortfolio,
    "risk_parity": RiskParityPortfolio,
    "erc": RiskParityPortfolio,
    "kelly": KellyPortfolio,
}


def make_portfolio(method: str, **kwargs) -> BasePortfolio:
    """
    Factory function for portfolio construction methods.

    Args:
        method : Portfolio method name (see _METHOD_MAP for valid names).
        **kwargs : Constructor arguments passed to the portfolio class.

    Returns:
        Instantiated portfolio object.

    Example:
        portfolio = make_portfolio('hrp')
        weights = portfolio.fit(returns_df)
    """
    cls = _METHOD_MAP.get(method.lower())
    if cls is None:
        raise ValueError(
            f"Unknown method '{method}'. Valid: {sorted(_METHOD_MAP.keys())}"
        )
    return cls(**kwargs)
