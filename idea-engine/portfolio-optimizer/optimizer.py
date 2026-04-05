"""
portfolio-optimizer/optimizer.py

Core portfolio optimiser implementing multiple strategies:
  - Mean-variance (Markowitz)
  - Minimum variance
  - Maximum Sharpe ratio
  - Risk parity (equal risk contribution)
  - Maximum diversification
  - Black-Litterman
  - Hierarchical Risk Parity (via hrp.py)
  - Robust mean-variance (ellipsoidal uncertainty set)
  - Genome portfolio optimisation: optimal allocation across strategy genomes

All weight vectors are checked to ensure they sum to 1.0 (within 1e-6).
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds

from portfolio_optimizer.hrp import HierarchicalRiskParity, HRPResult


# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OptimisationResult:
    """
    Result of a single portfolio optimisation run.

    Attributes
    ----------
    method : str
        Name of the optimisation method.
    weights : np.ndarray
        Optimal asset weights (sums to 1.0).
    asset_names : list[str]
        Asset names corresponding to each weight.
    expected_return : float
        Expected annual portfolio return.
    expected_volatility : float
        Expected annual portfolio volatility.
    sharpe_ratio : float
        Expected Sharpe ratio.
    metadata : dict
        Method-specific diagnostics (iterations, convergence, etc.).
    """

    method: str
    weights: np.ndarray
    asset_names: list[str]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_series(self) -> pd.Series:
        return pd.Series(self.weights, index=self.asset_names, name=self.method)

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.asset_names, self.weights.tolist()))

    def summary(self) -> str:
        lines = [
            f"Method          : {self.method}",
            f"Expected Return : {self.expected_return:.4f}",
            f"Expected Vol    : {self.expected_volatility:.4f}",
            f"Sharpe Ratio    : {self.sharpe_ratio:.4f}",
            "Weights:",
        ]
        for name, w in zip(self.asset_names, self.weights):
            lines.append(f"  {name:<20} {w:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core optimiser
# ---------------------------------------------------------------------------


class PortfolioOptimizer:
    """
    Multi-strategy portfolio weight optimiser.

    Parameters
    ----------
    db_path : str, optional
        Path to the SQLite database for persisting allocation results.
    rf : float
        Risk-free rate per bar (used in Sharpe calculations).  Default 0.0.
    annualisation : int
        Bars per year for annualising statistics.  Default 252.
    allow_short : bool
        Allow negative weights (short selling).  Default False.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> returns = pd.DataFrame(rng.normal(0.0005, 0.012, (500, 5)),
    ...                        columns=list("ABCDE"))
    >>> opt = PortfolioOptimizer()
    >>> result = opt.maximum_sharpe_portfolio(
    ...     returns.mean().values, returns.cov().values
    ... )
    >>> print(result.summary())
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        rf: float = 0.0,
        annualisation: int = 252,
        allow_short: bool = False,
    ) -> None:
        self.db_path = db_path
        self.rf = rf
        self.annualisation = annualisation
        self.allow_short = allow_short
        self._results: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Mean-variance optimisation
    # ------------------------------------------------------------------

    def mean_variance_optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 1.0,
        asset_names: Optional[list[str]] = None,
    ) -> OptimisationResult:
        """
        Maximise the risk-adjusted utility: U = μᵀw - (λ/2) wᵀΣw.

        Parameters
        ----------
        expected_returns : np.ndarray
            Vector of expected returns (per bar).
        cov_matrix : np.ndarray
            Asset covariance matrix.
        risk_aversion : float
            Risk aversion coefficient λ.  Higher values → lower risk.
        asset_names : list[str], optional
            Names for each asset.

        Returns
        -------
        OptimisationResult
        """
        n = len(expected_returns)
        assets = asset_names or [f"asset_{i}" for i in range(n)]

        def neg_utility(w: np.ndarray) -> float:
            return float(-(expected_returns @ w - 0.5 * risk_aversion * w @ cov_matrix @ w))

        def neg_utility_grad(w: np.ndarray) -> np.ndarray:
            return -(expected_returns - risk_aversion * cov_matrix @ w)

        result = self._run_optimizer(
            n,
            neg_utility,
            neg_utility_grad,
            method="mean_variance",
        )

        w = result.x
        return self._build_result("mean_variance", w, assets, expected_returns, cov_matrix)

    def minimum_variance_portfolio(
        self,
        cov_matrix: np.ndarray,
        asset_names: Optional[list[str]] = None,
    ) -> OptimisationResult:
        """
        Find the portfolio with minimum variance (no return requirement).

        Parameters
        ----------
        cov_matrix : np.ndarray
            Asset covariance matrix.
        asset_names : list[str], optional
            Asset names.

        Returns
        -------
        OptimisationResult
        """
        n = cov_matrix.shape[0]
        assets = asset_names or [f"asset_{i}" for i in range(n)]

        def portfolio_variance(w: np.ndarray) -> float:
            return float(w @ cov_matrix @ w)

        def variance_grad(w: np.ndarray) -> np.ndarray:
            return 2.0 * cov_matrix @ w

        result = self._run_optimizer(n, portfolio_variance, variance_grad, "min_variance")
        w = result.x
        mu = np.zeros(n)
        return self._build_result("minimum_variance", w, assets, mu, cov_matrix)

    def maximum_sharpe_portfolio(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        rf: Optional[float] = None,
        asset_names: Optional[list[str]] = None,
    ) -> OptimisationResult:
        """
        Maximise the Sharpe ratio.

        Uses the Tobin separation approach: maximise (μ - rf)ᵀw / sqrt(wᵀΣw).

        Parameters
        ----------
        expected_returns : np.ndarray
            Expected per-bar returns.
        cov_matrix : np.ndarray
            Asset covariance matrix.
        rf : float, optional
            Per-bar risk-free rate.  Uses instance default if None.
        asset_names : list[str], optional
            Asset names.

        Returns
        -------
        OptimisationResult
        """
        rf_ = rf if rf is not None else self.rf
        n = len(expected_returns)
        assets = asset_names or [f"asset_{i}" for i in range(n)]
        excess = expected_returns - rf_

        def neg_sharpe(w: np.ndarray) -> float:
            port_ret = float(excess @ w)
            port_vol = float(np.sqrt(w @ cov_matrix @ w + 1e-10))
            return -port_ret / port_vol

        result = self._run_optimizer(n, neg_sharpe, None, "max_sharpe")
        w = result.x
        return self._build_result("maximum_sharpe", w, assets, expected_returns, cov_matrix)

    # ------------------------------------------------------------------
    # Risk parity
    # ------------------------------------------------------------------

    def risk_parity_portfolio(
        self,
        cov_matrix: np.ndarray,
        asset_names: Optional[list[str]] = None,
    ) -> OptimisationResult:
        """
        Equal Risk Contribution (ERC / Risk Parity) portfolio.

        Each asset contributes equally to the total portfolio variance:
            RC_i = w_i * (Σw)_i / wᵀΣw  for all i.

        Solved by minimising the sum of squared differences between
        actual risk contributions and the target 1/n contribution.

        Parameters
        ----------
        cov_matrix : np.ndarray
            Asset covariance matrix.
        asset_names : list[str], optional
            Asset names.

        Returns
        -------
        OptimisationResult
        """
        n = cov_matrix.shape[0]
        assets = asset_names or [f"asset_{i}" for i in range(n)]
        target = 1.0 / n

        def erc_objective(w: np.ndarray) -> float:
            sigma2 = float(w @ cov_matrix @ w)
            if sigma2 < 1e-12:
                return 0.0
            marginal = cov_matrix @ w
            rc = w * marginal / sigma2
            return float(np.sum((rc - target) ** 2))

        # Use positive weights only (no shorting for risk parity)
        bounds = Bounds(lb=1e-4, ub=1.0)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        w0 = np.ones(n) / n

        res = minimize(
            erc_objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 2000},
        )
        w = self._normalise(res.x)
        return self._build_result("risk_parity", w, assets, np.zeros(n), cov_matrix)

    # ------------------------------------------------------------------
    # Maximum diversification
    # ------------------------------------------------------------------

    def max_diversification_portfolio(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        asset_names: Optional[list[str]] = None,
    ) -> OptimisationResult:
        """
        Maximise the diversification ratio: weighted-average volatility
        divided by portfolio volatility.

        DR = (wᵀσ) / sqrt(wᵀΣw)

        Parameters
        ----------
        expected_returns : np.ndarray
            Expected per-bar returns (used only for result statistics).
        cov_matrix : np.ndarray
            Asset covariance matrix.
        asset_names : list[str], optional
            Asset names.

        Returns
        -------
        OptimisationResult
        """
        n = cov_matrix.shape[0]
        assets = asset_names or [f"asset_{i}" for i in range(n)]
        asset_vols = np.sqrt(np.diag(cov_matrix))

        def neg_dr(w: np.ndarray) -> float:
            port_vol = float(np.sqrt(w @ cov_matrix @ w + 1e-12))
            weighted_avg_vol = float(w @ asset_vols)
            return -weighted_avg_vol / port_vol

        result = self._run_optimizer(n, neg_dr, None, "max_diversification")
        w = result.x
        return self._build_result(
            "max_diversification", w, assets, expected_returns, cov_matrix
        )

    # ------------------------------------------------------------------
    # Black-Litterman
    # ------------------------------------------------------------------

    def black_litterman(
        self,
        prior_returns: np.ndarray,
        cov_matrix: np.ndarray,
        views: np.ndarray,
        view_confidence: np.ndarray,
        view_matrix: Optional[np.ndarray] = None,
        tau: float = 0.025,
        asset_names: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, OptimisationResult]:
        """
        Black-Litterman model: blends equilibrium (prior) expected returns
        with investor views to produce a posterior expected-return vector,
        then optimises the maximum-Sharpe portfolio on that posterior.

        Parameters
        ----------
        prior_returns : np.ndarray
            Equilibrium (implied) expected returns (e.g. from CAPM).
        cov_matrix : np.ndarray
            Asset covariance matrix.
        views : np.ndarray
            View excess-return values (one per view).
        view_confidence : np.ndarray
            Diagonal of the view uncertainty matrix Ω.  Smaller = more
            confident.
        view_matrix : np.ndarray, optional
            Pick matrix P (k × n) mapping assets to views.  Defaults to
            identity (each view corresponds to one asset's absolute return).
        tau : float
            Scalar controlling the weight on the prior (typically 0.01-0.05).
        asset_names : list[str], optional
            Asset names.

        Returns
        -------
        (posterior_returns, OptimisationResult)
            Posterior expected-return vector and the optimal portfolio.
        """
        n = len(prior_returns)
        assets = asset_names or [f"asset_{i}" for i in range(n)]

        if view_matrix is None:
            k = len(views)
            P = np.eye(n)[:k]  # absolute views on first k assets
        else:
            P = np.asarray(view_matrix, dtype=float)

        Q = np.asarray(views, dtype=float)
        Omega = np.diag(np.asarray(view_confidence, dtype=float))

        # Black-Litterman formula
        tau_sigma = tau * cov_matrix
        # Posterior variance
        inv_tau_sigma = np.linalg.inv(tau_sigma)
        inv_omega = np.linalg.inv(Omega)
        post_cov_inv = inv_tau_sigma + P.T @ inv_omega @ P
        post_cov = np.linalg.inv(post_cov_inv)

        # Posterior mean
        post_mean = post_cov @ (inv_tau_sigma @ prior_returns + P.T @ inv_omega @ Q)

        # Optimise maximum Sharpe on posterior
        opt_result = self.maximum_sharpe_portfolio(post_mean, cov_matrix, asset_names=assets)
        opt_result.method = "black_litterman"

        return post_mean, opt_result

    # ------------------------------------------------------------------
    # Hierarchical Risk Parity
    # ------------------------------------------------------------------

    def hierarchical_risk_parity(
        self,
        returns_df: pd.DataFrame,
    ) -> OptimisationResult:
        """
        Hierarchical Risk Parity (López de Prado 2016).

        Parameters
        ----------
        returns_df : pd.DataFrame
            Asset returns (each column = one asset).

        Returns
        -------
        OptimisationResult
        """
        hrp = HierarchicalRiskParity()
        hrp_result = hrp.fit(returns_df)

        assets = list(returns_df.columns)
        w = hrp_result.weights.reindex(assets).fillna(0.0).values
        w = self._normalise(w)

        cov = returns_df.dropna().cov().values
        mu = returns_df.dropna().mean().values

        return self._build_result("hrp", w, assets, mu, cov)

    # ------------------------------------------------------------------
    # Robust optimisation
    # ------------------------------------------------------------------

    def robust_optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        uncertainty_set: str = "ellipsoidal",
        kappa: float = 1.0,
        asset_names: Optional[list[str]] = None,
    ) -> OptimisationResult:
        """
        Robust portfolio optimisation with uncertainty in expected returns.

        For the ellipsoidal set, the worst-case expected return is:
            min_μ μᵀw  s.t.  (μ - μ̂)ᵀ Σ⁻¹ (μ - μ̂) ≤ κ²

        The worst-case mean is: μ̂ - κ * sqrt(wᵀΣw) * e_norm, so the
        robust Sharpe denominator is inflated by κ√(wᵀΣw).

        Maximise: (μ̂ᵀw - κ√(wᵀΣw)) / √(wᵀΣw)

        Parameters
        ----------
        expected_returns : np.ndarray
            Estimated expected returns.
        cov_matrix : np.ndarray
            Asset covariance matrix.
        uncertainty_set : str
            Must be ``'ellipsoidal'``.  (Box set planned for future.)
        kappa : float
            Radius of the ellipsoidal uncertainty set.  Larger → more
            conservative (penalises estimation uncertainty more heavily).
        asset_names : list[str], optional
            Asset names.

        Returns
        -------
        OptimisationResult
        """
        if uncertainty_set != "ellipsoidal":
            raise NotImplementedError(f"Uncertainty set {uncertainty_set!r} not yet supported")

        n = len(expected_returns)
        assets = asset_names or [f"asset_{i}" for i in range(n)]

        def robust_neg_sharpe(w: np.ndarray) -> float:
            port_vol = float(np.sqrt(w @ cov_matrix @ w + 1e-10))
            port_ret = float(expected_returns @ w)
            # Worst-case return deduction
            penalty = kappa * port_vol
            return -(port_ret - penalty) / port_vol

        result = self._run_optimizer(n, robust_neg_sharpe, None, "robust")
        w = result.x
        return self._build_result("robust", w, assets, expected_returns, cov_matrix)

    # ------------------------------------------------------------------
    # Genome portfolio optimisation
    # ------------------------------------------------------------------

    def optimize_genome_portfolio(
        self,
        genome_ids: list[str],
        returns_matrix: pd.DataFrame,
        target_sharpe: float = 2.0,
        method: str = "max_sharpe",
        db_path: Optional[str] = None,
    ) -> OptimisationResult:
        """
        Find the optimal capital allocation across multiple trading-strategy
        genomes to achieve a target Sharpe ratio.

        Parameters
        ----------
        genome_ids : list[str]
            Genome identifiers (used as asset names in the result).
        returns_matrix : pd.DataFrame
            Simulated or historical returns for each genome, one column per
            genome, in the same order as ``genome_ids``.
        target_sharpe : float
            Desired Sharpe ratio.  The chosen method is used as-is; the target
            is informational and stored in metadata.
        method : str
            Optimisation method: ``'max_sharpe'``, ``'risk_parity'``,
            ``'hrp'``, or ``'min_variance'``.
        db_path : str, optional
            SQLite path for persisting the result.

        Returns
        -------
        OptimisationResult
        """
        clean = returns_matrix.reindex(columns=genome_ids).dropna()
        mu = clean.mean().values
        cov = clean.cov().values

        if method == "max_sharpe":
            result = self.maximum_sharpe_portfolio(mu, cov, asset_names=genome_ids)
        elif method == "risk_parity":
            result = self.risk_parity_portfolio(cov, asset_names=genome_ids)
        elif method == "hrp":
            result = self.hierarchical_risk_parity(clean)
            result.asset_names = genome_ids
        elif method == "min_variance":
            result = self.minimum_variance_portfolio(cov, asset_names=genome_ids)
        else:
            raise ValueError(f"Unknown method: {method!r}")

        result.method = f"genome_portfolio_{method}"
        result.metadata["genome_ids"] = genome_ids
        result.metadata["target_sharpe"] = target_sharpe
        result.metadata["achieved_sharpe"] = result.sharpe_ratio

        # Persist if db_path provided
        persist_path = db_path or self.db_path
        if persist_path:
            self._persist_allocation(result, persist_path, genome_ids)

        return result

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    def flush_to_db(self) -> int:
        """
        Write all buffered optimisation results to portfolio_allocations.

        Returns
        -------
        int
            Number of rows inserted.
        """
        if not self.db_path or not self._results:
            return 0

        import json

        rows = [
            (
                r["method"],
                json.dumps(r["genome_ids"]),
                json.dumps(r["weights"]),
                r.get("expected_sharpe"),
                r.get("expected_dd"),
            )
            for r in self._results
        ]

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO portfolio_allocations
                    (method, genome_ids, weights, expected_sharpe, expected_dd)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

        count = len(rows)
        self._results.clear()
        return count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_optimizer(
        self,
        n: int,
        objective: Any,
        grad: Any,
        method_label: str,
    ) -> Any:
        """Run SciPy SLSQP with standard portfolio constraints."""
        lb = -1.0 if self.allow_short else 0.0
        bounds = Bounds(lb=lb, ub=1.0)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        w0 = np.ones(n) / n

        result = minimize(
            objective,
            w0,
            jac=grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 2000, "disp": False},
        )
        return result

    def _build_result(
        self,
        method: str,
        weights: np.ndarray,
        asset_names: list[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> OptimisationResult:
        """Validate weights and compute portfolio statistics."""
        w = self._normalise(weights)
        self._check_weights(w)

        port_ret = float(expected_returns @ w) * self.annualisation
        port_vol = float(np.sqrt(w @ cov_matrix @ w)) * np.sqrt(self.annualisation)
        sharpe = (port_ret - self.rf * self.annualisation) / max(port_vol, 1e-10)

        result = OptimisationResult(
            method=method,
            weights=w,
            asset_names=asset_names,
            expected_return=port_ret,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
        )

        # Buffer for DB
        self._results.append(
            {
                "method": method,
                "genome_ids": asset_names,
                "weights": dict(zip(asset_names, w.tolist())),
                "expected_sharpe": sharpe,
                "expected_dd": None,
            }
        )

        return result

    @staticmethod
    def _normalise(w: np.ndarray) -> np.ndarray:
        """Normalise weights to sum to exactly 1.0."""
        w = np.asarray(w, dtype=float)
        total = w.sum()
        if total == 0:
            n = len(w)
            return np.ones(n) / n
        return w / total

    @staticmethod
    def _check_weights(weights: np.ndarray, tol: float = 1e-6) -> None:
        """Raise ValueError if weights do not sum to 1.0."""
        total = float(np.sum(weights))
        if abs(total - 1.0) > tol:
            raise ValueError(
                f"Portfolio weights must sum to 1.0; got {total:.8f}"
            )

    def _persist_allocation(
        self,
        result: OptimisationResult,
        db_path: str,
        genome_ids: list[str],
    ) -> None:
        """Write a single optimisation result to portfolio_allocations."""
        import json

        weights_dict = dict(zip(result.asset_names, result.weights.tolist()))
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO portfolio_allocations
                    (method, genome_ids, weights, expected_sharpe, expected_dd)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    result.method,
                    json.dumps(genome_ids),
                    json.dumps(weights_dict),
                    result.sharpe_ratio,
                    None,
                ),
            )
            conn.commit()
