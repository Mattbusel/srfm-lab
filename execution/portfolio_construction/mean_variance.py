# execution/portfolio_construction/mean_variance.py
# Markowitz mean-variance optimisation with Black-Litterman views.
#
# Implements:
#   - MeanVarianceOptimizer: max Sharpe, min variance, efficient frontier,
#     max return for vol target, Black-Litterman posterior
#   - PortfolioConstraints: per-asset, sector, leverage, turnover limits

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize


# ---------------------------------------------------------------------------
# PortfolioConstraints
# ---------------------------------------------------------------------------


@dataclass
class PortfolioConstraints:
    """
    Container for portfolio-level constraints used in optimisation.

    Attributes
    ----------
    max_weight : float
        Maximum weight per individual asset.  Default 0.30 (30%).
    min_weight : float
        Minimum weight per individual asset.  For long-only portfolios this
        should be >= 0.  Set to a negative value for long/short.
    max_sector_weight : dict
        Mapping from sector label to maximum aggregate weight.
        E.g. {"crypto": 0.40, "equity": 0.60}.
    asset_sector_map : dict
        Mapping from asset index (int) to sector label.
    max_gross_leverage : float
        Maximum sum of absolute weights.  1.0 = long-only, 2.0 = 130/30, etc.
    max_turnover : float
        Maximum one-way portfolio turnover per rebalance, expressed as a
        fraction of NAV.  E.g. 0.20 means at most 20% of NAV can be traded.
    long_only : bool
        If True, all weights >= 0.  If False, short weights are permitted
        down to min_weight.
    current_weights : np.ndarray, optional
        Current portfolio weights used when enforcing turnover constraints.
    """

    max_weight: float = 0.30
    min_weight: float = 0.0
    max_sector_weight: Dict[str, float] = field(default_factory=dict)
    asset_sector_map: Dict[int, str] = field(default_factory=dict)
    max_gross_leverage: float = 1.0
    max_turnover: float = 1.0  # 1.0 = unconstrained by default
    long_only: bool = True
    current_weights: Optional[np.ndarray] = None

    def build_bounds(self, n: int) -> List[Tuple[float, float]]:
        """Return a list of (min, max) tuples for scipy.optimize."""
        lo = 0.0 if self.long_only else self.min_weight
        hi = self.max_weight
        return [(lo, hi)] * n

    def build_scipy_constraints(
        self, n: int, mu: Optional[np.ndarray] = None
    ) -> List[dict]:
        """
        Build scipy constraint dicts.

        Always includes sum-to-1 (for long-only) or sum-to-gross-leverage
        (for long/short) constraint.

        Sector constraints are added when asset_sector_map is populated.
        Turnover constraint is added when current_weights is set.
        """
        constraints: List[dict] = []

        # Full-investment constraint.
        constraints.append({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})

        # Gross leverage constraint (redundant for long-only but explicit).
        if not self.long_only:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: self.max_gross_leverage - np.sum(np.abs(w)),
                }
            )

        # Per-sector constraints.
        for sector, sector_max in self.max_sector_weight.items():
            sector_indices = [
                i for i, s in self.asset_sector_map.items() if s == sector
            ]
            if not sector_indices:
                continue

            def _sector_constraint(w: np.ndarray, idx=sector_indices, cap=sector_max) -> float:
                return cap - np.sum(w[idx])

            constraints.append({"type": "ineq", "fun": _sector_constraint})

        # Turnover constraint.
        if self.current_weights is not None and self.max_turnover < 1.0:
            cw = np.asarray(self.current_weights, dtype=float)

            def _turnover_constraint(w: np.ndarray) -> float:
                return self.max_turnover - 0.5 * np.sum(np.abs(w - cw))

            constraints.append({"type": "ineq", "fun": _turnover_constraint})

        return constraints


# ---------------------------------------------------------------------------
# MeanVarianceOptimizer
# ---------------------------------------------------------------------------


class MeanVarianceOptimizer:
    """
    Markowitz mean-variance portfolio optimisation.

    Supports:
    - Maximum Sharpe ratio (via auxiliary variable trick)
    - Global minimum variance
    - Efficient frontier sweep
    - Maximum return for a given vol target
    - Black-Litterman posterior mean and covariance
    """

    # ------------------------------------------------------------------
    # Maximum Sharpe ratio
    # ------------------------------------------------------------------

    def max_sharpe(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        rf: float = 0.0,
        bounds: Tuple[float, float] = (0.0, 0.30),
        constraints: Optional[PortfolioConstraints] = None,
    ) -> np.ndarray:
        """
        Compute the maximum Sharpe ratio portfolio.

        Uses the Markowitz auxiliary-variable trick (Tobin separation):
            Let z = y / (y^T * (mu - rf))  where y are unnormalised weights.
            Then min z^T * Sigma * z  s.t.  y^T * (mu - rf) = 1,  y >= 0.

        For the simple (no extra constraints) case this is solved via the
        standard QP.  With extra constraints the weights are found by
        re-normalisation from SLSQP on the Sharpe directly.

        Parameters
        ----------
        mu : np.ndarray, shape (n,)
            Expected returns vector.
        cov : np.ndarray, shape (n, n)
            Covariance matrix.
        rf : float
            Risk-free rate.
        bounds : tuple (lo, hi)
            Per-asset weight bounds.
        constraints : PortfolioConstraints, optional
            If provided, additional constraints are applied.

        Returns
        -------
        np.ndarray, shape (n,)
        """
        mu = np.asarray(mu, dtype=float)
        cov = _ensure_psd(np.asarray(cov, dtype=float))
        n = len(mu)
        excess_mu = mu - rf

        if np.all(excess_mu <= 0):
            warnings.warn(
                "All expected returns <= risk-free rate.  "
                "Returning min-variance portfolio.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.min_variance(cov, bounds=bounds)

        # Initial guess: proportional to excess return.
        w0 = np.clip(excess_mu, 0.0, None)
        if w0.sum() == 0:
            w0 = np.full(n, 1.0 / n)
        else:
            w0 /= w0.sum()

        def _neg_sharpe(w: np.ndarray) -> float:
            port_ret = float(w @ mu) - rf
            port_vol = float(np.sqrt(w @ cov @ w))
            if port_vol <= 0:
                return 1e10
            return -port_ret / port_vol

        def _neg_sharpe_grad(w: np.ndarray) -> np.ndarray:
            eps = 1e-7
            grad = np.zeros(n)
            f0 = _neg_sharpe(w)
            for k in range(n):
                w_up = w.copy()
                w_up[k] += eps
                grad[k] = (_neg_sharpe(w_up) - f0) / eps
            return grad

        if constraints is not None:
            scipy_constraints = constraints.build_scipy_constraints(n, mu)
            bound_list = constraints.build_bounds(n)
        else:
            scipy_constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            bound_list = [bounds] * n

        result = minimize(
            _neg_sharpe,
            w0,
            jac=_neg_sharpe_grad,
            method="SLSQP",
            bounds=bound_list,
            constraints=scipy_constraints,
            options={"ftol": 1e-10, "maxiter": 2000, "disp": False},
        )

        if not result.success:
            warnings.warn(
                f"Max Sharpe optimisation did not converge: {result.message}.",
                RuntimeWarning,
                stacklevel=2,
            )

        weights = np.clip(result.x, bounds[0] if bounds[0] >= 0 else 0.0, bounds[1])
        weights /= weights.sum()
        return weights

    # ------------------------------------------------------------------
    # Global minimum variance
    # ------------------------------------------------------------------

    def min_variance(
        self,
        cov: np.ndarray,
        bounds: Tuple[float, float] = (0.01, 0.30),
        constraints: Optional[PortfolioConstraints] = None,
    ) -> np.ndarray:
        """
        Compute the global minimum variance portfolio.

        Solves:
            min  w^T * Sigma * w
            s.t. sum(w) = 1
                 bounds[0] <= w_i <= bounds[1]

        Parameters
        ----------
        cov : np.ndarray, shape (n, n)
        bounds : tuple (lo, hi)
        constraints : PortfolioConstraints, optional

        Returns
        -------
        np.ndarray, shape (n,)
        """
        cov = _ensure_psd(np.asarray(cov, dtype=float))
        n = cov.shape[0]
        w0 = np.full(n, 1.0 / n)

        def _portfolio_var(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        def _portfolio_var_grad(w: np.ndarray) -> np.ndarray:
            return 2.0 * (cov @ w)

        if constraints is not None:
            scipy_constraints = constraints.build_scipy_constraints(n)
            bound_list = constraints.build_bounds(n)
        else:
            scipy_constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            bound_list = [bounds] * n

        result = minimize(
            _portfolio_var,
            w0,
            jac=_portfolio_var_grad,
            method="SLSQP",
            bounds=bound_list,
            constraints=scipy_constraints,
            options={"ftol": 1e-12, "maxiter": 2000, "disp": False},
        )

        if not result.success:
            warnings.warn(
                f"Min variance optimisation did not converge: {result.message}.",
                RuntimeWarning,
                stacklevel=2,
            )

        weights = np.clip(result.x, bounds[0], bounds[1])
        weights /= weights.sum()
        return weights

    # ------------------------------------------------------------------
    # Efficient frontier
    # ------------------------------------------------------------------

    def efficient_frontier(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        n_points: int = 50,
        bounds: Tuple[float, float] = (0.0, 0.30),
    ) -> List[Tuple[float, float, np.ndarray]]:
        """
        Trace the efficient frontier.

        Sweeps from the minimum variance portfolio up to the maximum
        return portfolio, solving min variance at each target return level.

        Returns a list of (expected_return, expected_vol, weights) tuples,
        sorted in ascending order of expected return.  Only portfolios that
        lie on the efficient (non-dominated) frontier are returned --
        specifically those with expected return >= min-variance portfolio.

        Parameters
        ----------
        mu : np.ndarray, shape (n,)
        cov : np.ndarray, shape (n, n)
        n_points : int
            Number of frontier points to compute.
        bounds : tuple (lo, hi)

        Returns
        -------
        list of (ret: float, vol: float, weights: np.ndarray)
        """
        mu = np.asarray(mu, dtype=float)
        cov = _ensure_psd(np.asarray(cov, dtype=float))
        n = len(mu)

        # Bounds on target return: from min-variance expected return to max.
        w_mv = self.min_variance(cov, bounds=bounds)
        ret_min = float(w_mv @ mu)

        # Max return portfolio: put max weight on highest-return assets.
        bound_list = [bounds] * n
        ret_max = float(np.sum([bounds[1]] * n) * mu.max())
        # More precisely: solve unconstrained max return -- just load up on
        # the highest return asset within bounds.
        sorted_idx = np.argsort(mu)[::-1]
        w_max = np.full(n, bounds[0])
        remaining = 1.0 - bounds[0] * n
        for idx in sorted_idx:
            add = min(bounds[1] - bounds[0], remaining)
            w_max[idx] += add
            remaining -= add
            if remaining <= 1e-10:
                break
        ret_max = float(w_max @ mu)

        if ret_max <= ret_min:
            # Edge case: all assets have similar returns.
            return [(ret_min, float(np.sqrt(w_mv @ cov @ w_mv)), w_mv.copy())]

        target_returns = np.linspace(ret_min, ret_max, n_points)
        frontier: List[Tuple[float, float, np.ndarray]] = []

        for target_ret in target_returns:
            weights = self._min_var_for_return(mu, cov, target_ret, bounds)
            if weights is None:
                continue
            actual_ret = float(weights @ mu)
            actual_vol = float(np.sqrt(weights @ cov @ weights))
            frontier.append((actual_ret, actual_vol, weights))

        # Deduplicate and sort.
        frontier.sort(key=lambda x: x[0])
        # Filter dominated points (vol should be non-decreasing after the
        # min-variance point; keep all since they are by construction efficient).
        return frontier

    def _min_var_for_return(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        target_ret: float,
        bounds: Tuple[float, float],
    ) -> Optional[np.ndarray]:
        """
        Solve for min variance subject to expected return >= target_ret.
        """
        n = len(mu)
        w0 = np.full(n, 1.0 / n)

        def _portfolio_var(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        def _portfolio_var_grad(w: np.ndarray) -> np.ndarray:
            return 2.0 * (cov @ w)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w: float(w @ mu) - target_ret},
        ]
        bound_list = [bounds] * n

        result = minimize(
            _portfolio_var,
            w0,
            jac=_portfolio_var_grad,
            method="SLSQP",
            bounds=bound_list,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 1000, "disp": False},
        )

        if not result.success:
            return None

        weights = np.clip(result.x, bounds[0], bounds[1])
        if weights.sum() > 0:
            weights /= weights.sum()
        return weights

    # ------------------------------------------------------------------
    # Max return for vol target
    # ------------------------------------------------------------------

    def max_return_for_vol(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        target_vol: float,
        bounds: Tuple[float, float] = (0.0, 0.30),
    ) -> np.ndarray:
        """
        Find the portfolio that maximises expected return subject to
        portfolio volatility <= target_vol.

        Solves:
            max  w^T * mu
            s.t. w^T * Sigma * w <= target_vol^2
                 sum(w) = 1
                 bounds[0] <= w_i <= bounds[1]

        Parameters
        ----------
        mu : np.ndarray, shape (n,)
        cov : np.ndarray, shape (n, n)
        target_vol : float
            Maximum allowed annualised portfolio volatility.
        bounds : tuple (lo, hi)

        Returns
        -------
        np.ndarray, shape (n,)
        """
        mu = np.asarray(mu, dtype=float)
        cov = _ensure_psd(np.asarray(cov, dtype=float))
        n = len(mu)
        w0 = np.full(n, 1.0 / n)

        def _neg_return(w: np.ndarray) -> float:
            return -float(w @ mu)

        def _neg_return_grad(w: np.ndarray) -> np.ndarray:
            return -mu

        def _vol_constraint(w: np.ndarray) -> float:
            return target_vol ** 2 - float(w @ cov @ w)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": _vol_constraint},
        ]
        bound_list = [bounds] * n

        result = minimize(
            _neg_return,
            w0,
            jac=_neg_return_grad,
            method="SLSQP",
            bounds=bound_list,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 2000, "disp": False},
        )

        if not result.success:
            warnings.warn(
                f"Max return for vol optimisation did not converge: {result.message}.",
                RuntimeWarning,
                stacklevel=2,
            )

        weights = np.clip(result.x, bounds[0], bounds[1])
        if weights.sum() > 0:
            weights /= weights.sum()
        return weights

    # ------------------------------------------------------------------
    # Black-Litterman
    # ------------------------------------------------------------------

    def black_litterman(
        self,
        mu_prior: np.ndarray,
        cov: np.ndarray,
        views: List[Tuple[np.ndarray, float]],
        view_confidences: np.ndarray,
        tau: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Black-Litterman posterior mean and covariance.

        Incorporates investor views via the BL formula:

            mu_BL = M^{-1} * v
            Sigma_BL = M^{-1}

        where:
            M = (tau * Sigma)^{-1} + P^T * Omega^{-1} * P
            v = (tau * Sigma)^{-1} * mu_prior + P^T * Omega^{-1} * q

        P is the K x n views matrix; each row is a view portfolio (e.g.
        [0, 0, 1, -1, 0] for a relative view on assets 2 vs 3).
        q is the K-vector of expected view returns.
        Omega is the K x K diagonal confidence matrix, where:
            Omega_kk = (1 - view_confidences[k]) / view_confidences[k] * P_k Sigma P_k^T

        Parameters
        ----------
        mu_prior : np.ndarray, shape (n,)
            Prior expected returns (e.g. equilibrium / CAPM implied).
        cov : np.ndarray, shape (n, n)
            Covariance matrix.
        views : list of (view_vector, view_return)
            Each view_vector is an np.ndarray of shape (n,).
        view_confidences : np.ndarray, shape (K,)
            Confidence in each view in (0, 1).  1 = certainty, 0 = no view.
        tau : float
            Scaling factor for prior uncertainty.  Typically 0.05 -- 0.10.

        Returns
        -------
        mu_bl : np.ndarray, shape (n,)
            Posterior expected returns.
        cov_bl : np.ndarray, shape (n, n)
            Posterior covariance matrix.
        """
        mu_prior = np.asarray(mu_prior, dtype=float)
        cov = _ensure_psd(np.asarray(cov, dtype=float))
        n = len(mu_prior)
        K = len(views)

        if K == 0:
            # No views: return prior unchanged.
            return mu_prior.copy(), cov.copy()

        view_confidences = np.asarray(view_confidences, dtype=float)
        view_confidences = np.clip(view_confidences, 1e-6, 1.0 - 1e-6)

        # Build P (K x n) and q (K,).
        P = np.zeros((K, n))
        q = np.zeros(K)
        for k, (pv, pr) in enumerate(views):
            P[k, :] = np.asarray(pv, dtype=float)
            q[k] = float(pr)

        # Build Omega: diagonal confidence-weighted uncertainty.
        # Omega_kk = ((1 - c_k) / c_k) * P_k * Sigma * P_k^T
        tau_cov = tau * cov
        Omega_diag = np.zeros(K)
        for k in range(K):
            p_k = P[k, :]
            variance_view = float(p_k @ tau_cov @ p_k)
            confidence_ratio = (1.0 - view_confidences[k]) / view_confidences[k]
            Omega_diag[k] = confidence_ratio * variance_view + 1e-12
        Omega = np.diag(Omega_diag)
        Omega_inv = np.diag(1.0 / Omega_diag)

        # Prior precision matrix.
        tau_cov_inv = np.linalg.inv(tau_cov + np.eye(n) * 1e-12)

        # Posterior precision matrix M.
        M = tau_cov_inv + P.T @ Omega_inv @ P

        # Posterior mean numerator v.
        v = tau_cov_inv @ mu_prior + P.T @ Omega_inv @ q

        # Solve M * mu_bl = v.
        try:
            M_inv = np.linalg.inv(M + np.eye(n) * 1e-12)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)

        mu_bl = M_inv @ v
        # Posterior covariance = parameter uncertainty + sampling uncertainty.
        cov_bl = _ensure_psd(M_inv + cov)

        return mu_bl, cov_bl

    # ------------------------------------------------------------------
    # Convenience: compute Sharpe for a portfolio
    # ------------------------------------------------------------------

    @staticmethod
    def portfolio_sharpe(
        weights: np.ndarray,
        mu: np.ndarray,
        cov: np.ndarray,
        rf: float = 0.0,
    ) -> float:
        """Compute ex-ante Sharpe ratio for a given weight vector."""
        weights = np.asarray(weights, dtype=float)
        mu = np.asarray(mu, dtype=float)
        cov = np.asarray(cov, dtype=float)
        port_ret = float(weights @ mu) - rf
        port_vol = float(np.sqrt(weights @ cov @ weights))
        if port_vol <= 0:
            return 0.0
        return port_ret / port_vol

    @staticmethod
    def portfolio_vol(weights: np.ndarray, cov: np.ndarray) -> float:
        """Compute ex-ante annualised portfolio volatility."""
        weights = np.asarray(weights, dtype=float)
        cov = np.asarray(cov, dtype=float)
        return float(np.sqrt(weights @ cov @ weights))

    @staticmethod
    def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
        """Compute ex-ante expected portfolio return."""
        return float(np.asarray(weights) @ np.asarray(mu))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _ensure_psd(cov: np.ndarray, min_eigenvalue: float = 1e-8) -> np.ndarray:
    """Project onto the PSD cone by clipping negative eigenvalues."""
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, min_eigenvalue, None)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
