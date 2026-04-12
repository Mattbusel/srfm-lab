"""
AETERNUS Real-Time Execution Layer (RTEL)
portfolio_optimizer.py — Portfolio optimization and execution planning

Provides:
- Mean-variance optimization (MVO)
- Equal Risk Contribution (ERC)
- Black-Litterman model
- Transaction cost-aware rebalancing
- Kelly criterion position sizing
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Covariance estimators
# ---------------------------------------------------------------------------

class LedoitWolfShrinkage:
    """Ledoit-Wolf analytical shrinkage estimator."""

    @staticmethod
    def estimate(returns: np.ndarray) -> np.ndarray:
        """
        Returns covariance matrix with L-W shrinkage.
        returns: [T × N] array of returns.
        """
        T, N = returns.shape
        if T < 2 or N < 1:
            return np.eye(N)

        X     = returns - returns.mean(axis=0)
        S     = (X.T @ X) / T
        mu_S  = np.trace(S) / N
        F     = mu_S * np.eye(N)

        # Analytical shrinkage intensity
        delta2 = np.sum((X ** 2).T @ (X ** 2)) / T**2 - np.sum(S**2) / T
        beta   = delta2 / (np.sum((S - F)**2) + _EPS)
        alpha  = max(0.0, min(1.0, beta))

        return alpha * F + (1.0 - alpha) * S


class EWMACovariance:
    """Exponentially weighted covariance matrix."""

    def __init__(self, n_assets: int, alpha: float = 0.06):
        self.alpha = alpha
        self._mean = np.zeros(n_assets)
        self._cov  = np.eye(n_assets) * 1e-4
        self._n    = 0

    def update(self, returns: np.ndarray) -> None:
        a = self.alpha
        self._mean = a * returns + (1-a) * self._mean
        diff = returns - self._mean
        self._cov  = (1-a) * self._cov + a * np.outer(diff, diff)
        self._n   += 1

    @property
    def covariance(self) -> np.ndarray:
        return self._cov.copy()

    @property
    def correlation(self) -> np.ndarray:
        std = np.sqrt(np.diag(self._cov))
        outer_std = np.outer(std, std)
        with np.errstate(invalid='ignore'):
            corr = np.where(outer_std > _EPS, self._cov / outer_std, 0.0)
        np.fill_diagonal(corr, 1.0)
        return corr


# ---------------------------------------------------------------------------
# Mean-variance optimizer
# ---------------------------------------------------------------------------

class MVOptimizer:
    """
    Markowitz mean-variance optimization.
    Uses analytical solution for max Sharpe and min variance.
    """

    def __init__(self, n_assets: int, risk_free_rate: float = 0.0):
        self.n         = n_assets
        self.rfr       = risk_free_rate

    def min_variance(self, cov: np.ndarray) -> np.ndarray:
        """Global minimum variance portfolio."""
        n = self.n
        ones = np.ones(n)
        try:
            inv_cov = np.linalg.inv(cov + _EPS * np.eye(n))
        except np.linalg.LinAlgError:
            return ones / n
        denom = ones @ inv_cov @ ones
        if denom < _EPS:
            return ones / n
        w = (inv_cov @ ones) / denom
        return w.clip(0, 1)

    def max_sharpe(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Tangency (max Sharpe) portfolio."""
        n    = self.n
        ones = np.ones(n)
        excess = mu - self.rfr
        try:
            inv_cov = np.linalg.inv(cov + _EPS * np.eye(n))
        except np.linalg.LinAlgError:
            return ones / n
        z     = inv_cov @ excess
        z_sum = z.sum()
        if z_sum < _EPS:
            return ones / n
        w = z / z_sum
        # Long-only constraint
        w = w.clip(0, None)
        w_sum = w.sum()
        if w_sum < _EPS:
            return ones / n
        return w / w_sum

    def efficient_frontier(self, mu: np.ndarray, cov: np.ndarray,
                           n_points: int = 20) -> List[Tuple[float, float, np.ndarray]]:
        """
        Returns list of (return, vol, weights) on efficient frontier.
        """
        points = []
        # Range of target returns
        mu_min = mu.min()
        mu_max = mu.max()
        for target_ret in np.linspace(mu_min, mu_max, n_points):
            w = self._min_var_for_target(mu, cov, target_ret)
            p_ret = float(w @ mu)
            p_var = float(w @ cov @ w)
            p_vol = math.sqrt(max(0.0, p_var))
            points.append((p_ret, p_vol, w))
        return points

    def _min_var_for_target(self, mu: np.ndarray, cov: np.ndarray,
                             target_ret: float) -> np.ndarray:
        """Min variance portfolio for given target return (KKT analytical)."""
        n = self.n
        try:
            inv_cov = np.linalg.inv(cov + _EPS * np.eye(n))
        except np.linalg.LinAlgError:
            return np.ones(n) / n
        ones = np.ones(n)
        A = float(mu @ inv_cov @ mu)
        B = float(mu @ inv_cov @ ones)
        C = float(ones @ inv_cov @ ones)
        D = A * C - B * B
        if abs(D) < _EPS:
            return np.ones(n) / n
        lam1 = (C * target_ret - B) / D
        lam2 = (A - B * target_ret) / D
        w    = lam1 * (inv_cov @ mu) + lam2 * (inv_cov @ ones)
        # Long-only projection
        w    = w.clip(0, None)
        w_sum = w.sum()
        return w / w_sum if w_sum > _EPS else np.ones(n) / n


# ---------------------------------------------------------------------------
# Equal Risk Contribution
# ---------------------------------------------------------------------------

class ERCOptimizer:
    """
    Equal Risk Contribution (risk parity) portfolio.
    Iterative Newton-Raphson solver.
    """

    def __init__(self, n_assets: int, max_iter: int = 200, tol: float = 1e-8):
        self.n        = n_assets
        self.max_iter = max_iter
        self.tol      = tol

    def optimize(self, cov: np.ndarray,
                 risk_budget: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve for ERC weights.
        risk_budget: target risk contributions (default = equal = 1/N).
        """
        n = self.n
        if risk_budget is None:
            risk_budget = np.ones(n) / n
        else:
            risk_budget = np.array(risk_budget)
            risk_budget /= risk_budget.sum()

        # Initial guess: inverse vol
        diag = np.diag(cov)
        diag = np.maximum(diag, _EPS)
        w = 1.0 / np.sqrt(diag)
        w /= w.sum()

        for _ in range(self.max_iter):
            # Compute marginal risk contributions
            sigma_w = cov @ w
            p_var   = float(w @ sigma_w)
            if p_var < _EPS:
                break
            p_vol   = math.sqrt(p_var)
            mrc     = sigma_w / p_vol
            rc      = w * mrc
            # Total rc = p_vol; target rc = p_vol * risk_budget
            target_rc = p_vol * risk_budget

            # Update via scaling
            grad = rc - target_rc
            if np.linalg.norm(grad) < self.tol:
                break
            w_new = w * (target_rc / (rc + _EPS))
            w_new /= w_new.sum()

            if np.linalg.norm(w_new - w) < self.tol:
                w = w_new
                break
            w = w_new

        return w.clip(0, None) / (w.sum() + _EPS)


# ---------------------------------------------------------------------------
# Black-Litterman model
# ---------------------------------------------------------------------------

class BlackLitterman:
    """
    Black-Litterman model for incorporating analyst views.
    """

    def __init__(self, n_assets: int, tau: float = 0.05, risk_aversion: float = 2.5):
        self.n     = n_assets
        self.tau   = tau
        self.delta = risk_aversion

    def compute_equilibrium_returns(self, cov: np.ndarray,
                                     market_weights: np.ndarray) -> np.ndarray:
        """Pi = delta * Sigma * w_mkt"""
        return self.delta * cov @ market_weights

    def posterior(self,
                  cov: np.ndarray,
                  market_weights: np.ndarray,
                  P: np.ndarray,
                  Q: np.ndarray,
                  Omega: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior return estimates.
        P: [k × N] view matrix
        Q: [k] view returns
        Omega: [k × k] view uncertainty (default: proportional to P*Sigma*P')
        Returns (mu_post, Sigma_post)
        """
        Pi    = self.compute_equilibrium_returns(cov, market_weights)
        tau_S = self.tau * cov

        if Omega is None:
            Omega = np.diag(np.diag(P @ tau_S @ P.T)) * self.tau

        # BL formula
        inv_tau_S = np.linalg.inv(tau_S + _EPS * np.eye(self.n))
        inv_Omega = np.linalg.inv(Omega + _EPS * np.eye(len(Q)))
        M_inv     = inv_tau_S + P.T @ inv_Omega @ P
        try:
            M         = np.linalg.inv(M_inv)
        except np.linalg.LinAlgError:
            return Pi, cov

        mu_post    = M @ (inv_tau_S @ Pi + P.T @ inv_Omega @ Q)
        Sigma_post = cov + M
        return mu_post, Sigma_post


# ---------------------------------------------------------------------------
# Transaction cost model
# ---------------------------------------------------------------------------

class TransactionCostModel:
    """Models transaction costs for rebalancing decisions."""

    def __init__(self, spread_bps: float = 5.0, impact_coeff: float = 0.1,
                 fixed_cost: float = 0.0):
        self.spread_bps   = spread_bps
        self.impact_coeff = impact_coeff
        self.fixed_cost   = fixed_cost

    def cost_for_trade(self, trade_usd: float, adv_usd: float = 1e6) -> float:
        """Returns total dollar cost for a trade."""
        abs_trade  = abs(trade_usd)
        if abs_trade < _EPS:
            return 0.0
        spread_cost = abs_trade * self.spread_bps / 1e4
        impact_cost = abs_trade * self.impact_coeff * (abs_trade / max(adv_usd, 1.0))**0.5
        return spread_cost + impact_cost + self.fixed_cost

    def cost_for_rebalance(self, current_weights: np.ndarray,
                            target_weights: np.ndarray,
                            portfolio_value: float,
                            adv_usd: Optional[np.ndarray] = None) -> float:
        """Total cost to rebalance from current to target."""
        trades = (target_weights - current_weights) * portfolio_value
        n = len(trades)
        if adv_usd is None:
            adv_usd = np.full(n, 1e6)
        return sum(self.cost_for_trade(trades[i], adv_usd[i]) for i in range(n))

    def net_benefit(self, expected_alpha: float, cost: float) -> float:
        """Alpha improvement minus transaction cost."""
        return expected_alpha - cost


# ---------------------------------------------------------------------------
# Portfolio rebalancer
# ---------------------------------------------------------------------------

class PortfolioRebalancer:
    """
    Transaction cost-aware portfolio rebalancer.
    Computes optimal rebalancing trades given target weights.
    """

    def __init__(self, tc_model: TransactionCostModel,
                 min_rebalance_pct: float = 0.005,
                 max_turnover: float = 0.5):
        self.tc_model          = tc_model
        self.min_rebalance_pct = min_rebalance_pct
        self.max_turnover      = max_turnover

    def compute_trades(self,
                       current_weights: np.ndarray,
                       target_weights:  np.ndarray,
                       portfolio_value: float,
                       prices:          np.ndarray) -> np.ndarray:
        """
        Returns dollar amount to trade for each asset (+ = buy, - = sell).
        Applies minimum trade threshold and max turnover constraint.
        """
        diffs     = target_weights - current_weights
        threshold = self.min_rebalance_pct
        trades    = np.where(np.abs(diffs) > threshold, diffs, 0.0)

        # Max turnover constraint
        turnover = np.abs(trades).sum() / 2.0
        if turnover > self.max_turnover:
            scale = self.max_turnover / turnover
            trades *= scale

        return trades * portfolio_value

    def should_rebalance(self,
                          current_weights: np.ndarray,
                          target_weights:  np.ndarray,
                          portfolio_value: float,
                          expected_alpha_improvement: float,
                          adv_usd: Optional[np.ndarray] = None) -> bool:
        """Return True if rebalance is cost-effective."""
        cost = self.tc_model.cost_for_rebalance(
            current_weights, target_weights, portfolio_value, adv_usd)
        return self.tc_model.net_benefit(expected_alpha_improvement, cost) > 0


# ---------------------------------------------------------------------------
# Kelly position sizer
# ---------------------------------------------------------------------------

class KellyPositionSizer:
    """
    Kelly criterion-based position sizing.
    Supports fractional Kelly to reduce variance.
    """

    def __init__(self, kelly_fraction: float = 0.25, max_position: float = 0.25):
        self.kelly_fraction = kelly_fraction
        self.max_position   = max_position

    def size_from_moments(self, mu: float, sigma: float) -> float:
        """Kelly fraction for continuous returns."""
        if sigma < _EPS:
            return 0.0
        full_kelly = mu / (sigma * sigma)
        sized      = full_kelly * self.kelly_fraction
        return float(np.clip(sized, -self.max_position, self.max_position))

    def size_portfolio(self, signals: np.ndarray,
                       vols: np.ndarray,
                       corr: np.ndarray) -> np.ndarray:
        """
        Size full portfolio using multivariate Kelly.
        w ∝ Sigma^{-1} * mu, then scale by Kelly fraction.
        """
        n   = len(signals)
        cov = np.outer(vols, vols) * corr
        try:
            inv_cov = np.linalg.inv(cov + _EPS * np.eye(n))
        except np.linalg.LinAlgError:
            inv_cov = np.eye(n)
        full_kelly = inv_cov @ signals
        sized      = full_kelly * self.kelly_fraction
        # Clip each asset
        sized = np.clip(sized, -self.max_position, self.max_position)
        return sized


# ---------------------------------------------------------------------------
# PortfolioOptimizationEngine — top-level optimizer
# ---------------------------------------------------------------------------

class PortfolioOptimizationEngine:
    """
    Integrates all portfolio optimization components.
    Produces target weights from signals and market data.
    """

    def __init__(self, n_assets: int, method: str = "erc",
                 kelly_fraction: float = 0.25):
        self.n_assets = n_assets
        self.method   = method

        self.cov_estimator = EWMACovariance(n_assets)
        self.mvo           = MVOptimizer(n_assets)
        self.erc           = ERCOptimizer(n_assets)
        self.bl            = BlackLitterman(n_assets)
        self.kelly_sizer   = KellyPositionSizer(kelly_fraction)
        self.tc_model      = TransactionCostModel()
        self.rebalancer    = PortfolioRebalancer(self.tc_model)

        self._current_weights = np.ones(n_assets) / n_assets
        self._step = 0

    def update_returns(self, returns: np.ndarray) -> None:
        """Update covariance estimate with latest returns."""
        self.cov_estimator.update(returns)
        self._step += 1

    def compute_target_weights(self, signals: np.ndarray,
                                 prices: np.ndarray) -> np.ndarray:
        """
        Compute target portfolio weights from signals.
        signals: [N] array of signal values in [-1, +1]
        prices:  [N] array of current prices
        """
        cov = self.cov_estimator.covariance
        vols = np.sqrt(np.diag(cov))

        if self.method == "erc":
            base_weights = self.erc.optimize(cov)
            # Apply signal direction
            signals_clipped = np.clip(signals, -1.0, 1.0)
            weights = base_weights * (1.0 + signals_clipped * 0.5)

        elif self.method == "mvo":
            mu = signals * vols * 0.1  # rough expected return estimate
            weights = self.mvo.max_sharpe(mu, cov)

        elif self.method == "kelly":
            corr = self.cov_estimator.correlation
            weights = self.kelly_sizer.size_portfolio(signals, vols, corr)
            # Convert to long-only weights (long positive signals)
            weights = weights.clip(0, None)

        elif self.method == "min_var":
            weights = self.mvo.min_variance(cov)

        else:
            weights = np.ones(self.n_assets) / self.n_assets

        # Normalize
        w_sum = weights.sum()
        if w_sum > _EPS:
            weights = weights / w_sum
        else:
            weights = np.ones(self.n_assets) / self.n_assets

        return weights

    def rebalance(self, signals: np.ndarray, prices: np.ndarray,
                  portfolio_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute rebalancing trades.
        Returns (target_weights, trade_amounts).
        """
        target = self.compute_target_weights(signals, prices)
        trades = self.rebalancer.compute_trades(
            self._current_weights, target, portfolio_value, prices)
        # Update current weights (simulate fills)
        if portfolio_value > _EPS:
            self._current_weights = target.copy()
        return target, trades

    def diagnostics(self) -> dict:
        cov = self.cov_estimator.covariance
        vols = np.sqrt(np.diag(cov))
        return {
            "step":           self._step,
            "method":         self.method,
            "current_weights": self._current_weights.tolist(),
            "vol_estimates":  vols.tolist(),
            "portfolio_vol":  float(np.sqrt(
                self._current_weights @ cov @ self._current_weights)),
        }
