"""
research/portfolio_lab/risk.py

Portfolio risk analytics for SRFM-Lab.

All functions operate on numpy arrays or pandas DataFrames of returns.

PortfolioRiskAnalyzer wraps all analytics into a stateful class.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from research.portfolio_lab.construction import _estimate_covariance, _cov_to_corr


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Compute annualised portfolio volatility.

    Args:
        weights    : Asset weights, shape (N,).
        cov_matrix : Covariance matrix (already annualised), shape (N, N).

    Returns:
        Scalar annualised volatility (same units as cov_matrix diagonal).
    """
    w = np.asarray(weights, dtype=np.float64)
    return float(math.sqrt(max(float(w @ cov_matrix @ w), 0.0)))


def marginal_risk_contribution(
    weights: np.ndarray, cov_matrix: np.ndarray
) -> np.ndarray:
    """
    Marginal risk contribution of each asset.

    MRC_i = ∂σ_p / ∂w_i = (Σw)_i / σ_p

    Args:
        weights    : Shape (N,).
        cov_matrix : Shape (N, N).

    Returns:
        MRC vector, shape (N,).
    """
    w = np.asarray(weights, dtype=np.float64)
    sigma = portfolio_volatility(w, cov_matrix) + 1e-12
    return (cov_matrix @ w) / sigma


def component_risk_contribution(
    weights: np.ndarray, cov_matrix: np.ndarray
) -> np.ndarray:
    """
    Component (percentage) risk contribution of each asset.

    CRC_i = w_i * MRC_i / σ_p  (sums to 1.0)

    Args:
        weights    : Shape (N,).
        cov_matrix : Shape (N, N).

    Returns:
        CRC vector, shape (N,) — percentage contributions summing to 1.0.
    """
    w = np.asarray(weights, dtype=np.float64)
    sigma = portfolio_volatility(w, cov_matrix) + 1e-12
    mrc = marginal_risk_contribution(w, cov_matrix)
    crc = w * mrc / sigma
    return crc


def portfolio_var(
    weights: np.ndarray,
    returns: np.ndarray,
    confidence: float = 0.99,
    method: str = "historical",
    cov_matrix: Optional[np.ndarray] = None,
    ann_factor: int = 252,
) -> float:
    """
    Portfolio Value at Risk (VaR).

    Args:
        weights    : Asset weights, shape (N,).
        returns    : Returns matrix, shape (T, N).
        confidence : Confidence level (e.g. 0.99 for 99% VaR).
        method     : 'historical' | 'parametric' | 'cornish_fisher'.
        cov_matrix : Precomputed covariance (for parametric). If None, estimated.
        ann_factor : Annualisation factor.

    Returns:
        Daily VaR as a positive fraction.
    """
    w = np.asarray(weights, dtype=np.float64)
    R = np.asarray(returns, dtype=np.float64)
    port_returns = R @ w

    if method == "historical":
        var = float(-np.percentile(port_returns, (1.0 - confidence) * 100))
        return max(var, 0.0)

    elif method == "parametric":
        mu = float(port_returns.mean())
        sigma = float(port_returns.std()) + 1e-12
        from scipy.stats import norm
        z = float(norm.ppf(1.0 - confidence))
        return float(max(-(mu + z * sigma), 0.0))

    elif method == "cornish_fisher":
        # Cornish-Fisher expansion using skewness and excess kurtosis
        from scipy.stats import norm
        mu = float(port_returns.mean())
        sigma = float(port_returns.std()) + 1e-12
        skew = float(_skewness(port_returns))
        kurt = float(_excess_kurtosis(port_returns))
        alpha = 1.0 - confidence
        z_alpha = float(norm.ppf(alpha))
        # Adjusted quantile
        z_cf = (z_alpha
                + (z_alpha ** 2 - 1) * skew / 6.0
                + (z_alpha ** 3 - 3 * z_alpha) * kurt / 24.0
                - (2 * z_alpha ** 3 - 5 * z_alpha) * skew ** 2 / 36.0)
        return float(max(-(mu + z_cf * sigma), 0.0))

    else:
        raise ValueError(f"Unknown VaR method: {method}.")


def portfolio_cvar(
    weights: np.ndarray,
    returns: np.ndarray,
    confidence: float = 0.99,
) -> float:
    """
    Conditional VaR (Expected Shortfall) — average loss beyond VaR.

    Args:
        weights    : Asset weights, shape (N,).
        returns    : Returns matrix, shape (T, N).
        confidence : Confidence level.

    Returns:
        Daily CVaR as a positive fraction.
    """
    w = np.asarray(weights, dtype=np.float64)
    R = np.asarray(returns, dtype=np.float64)
    port_returns = R @ w

    cutoff = np.percentile(port_returns, (1.0 - confidence) * 100)
    tail = port_returns[port_returns <= cutoff]
    if len(tail) == 0:
        return float(-np.min(port_returns))
    return float(-np.mean(tail))


def portfolio_beta(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> float:
    """
    Portfolio beta vs. benchmark: Cov(r_p, r_b) / Var(r_b).

    Args:
        portfolio_returns  : 1-D array of portfolio returns.
        benchmark_returns  : 1-D array of benchmark returns.

    Returns:
        Beta scalar.
    """
    p = np.asarray(portfolio_returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)
    var_b = float(np.var(b)) + 1e-12
    cov_pb = float(np.cov(p, b)[0, 1])
    return cov_pb / var_b


def tracking_error(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    ann_factor: int = 252,
) -> float:
    """
    Annualised tracking error: std(r_p - r_b) * sqrt(ann_factor).

    Args:
        portfolio_returns  : 1-D array.
        benchmark_returns  : 1-D array.
        ann_factor         : Annualisation factor.

    Returns:
        Annualised tracking error.
    """
    p = np.asarray(portfolio_returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)
    diff = p - b
    return float(np.std(diff) * math.sqrt(ann_factor))


def information_ratio(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    ann_factor: int = 252,
) -> float:
    """
    Annualised information ratio: mean(r_p - r_b) / tracking_error.

    Args:
        portfolio_returns : 1-D array.
        benchmark_returns : 1-D array.
        ann_factor        : Annualisation factor.

    Returns:
        Information ratio scalar.
    """
    p = np.asarray(portfolio_returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)
    diff = p - b
    te = float(np.std(diff) * math.sqrt(ann_factor)) + 1e-12
    return float(np.mean(diff) * ann_factor / te)


def stress_test_portfolio(
    weights: np.ndarray,
    stress_scenarios: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Apply stress scenarios to the portfolio and return P&L for each.

    Args:
        weights          : Asset weights, shape (N,).
        stress_scenarios : Dict mapping scenario name to return shock vector (N,).
                           Each shock vector represents percentage return shocks
                           to apply simultaneously to all assets.

    Returns:
        Dict mapping scenario name to portfolio P&L (negative = loss).

    Example:
        scenarios = {
            '2020_crash': np.array([-0.30, -0.28, -0.15]),
            'rate_spike': np.array([-0.05, -0.08, +0.02]),
        }
        pnl = stress_test_portfolio(weights, scenarios)
    """
    w = np.asarray(weights, dtype=np.float64)
    results = {}
    for name, shocks in stress_scenarios.items():
        s = np.asarray(shocks, dtype=np.float64)
        if s.shape[0] != w.shape[0]:
            raise ValueError(
                f"Scenario '{name}' has {s.shape[0]} shocks but portfolio has {w.shape[0]} assets."
            )
        results[name] = float(w @ s)
    return results


# ---------------------------------------------------------------------------
# PortfolioRiskAnalyzer
# ---------------------------------------------------------------------------


class PortfolioRiskAnalyzer:
    """
    Comprehensive portfolio risk analytics.

    Args:
        ann_factor : Annualisation factor (252 for daily).
        cov_method : Default covariance estimation method.
    """

    def __init__(
        self,
        ann_factor: int = 252,
        cov_method: str = "ledoit_wolf",
    ) -> None:
        self.ann_factor = ann_factor
        self.cov_method = cov_method

    def covariance_estimation(
        self,
        returns: pd.DataFrame,
        method: Optional[str] = None,
    ) -> np.ndarray:
        """
        Estimate the covariance matrix.

        Args:
            returns : Returns DataFrame (T, N).
            method  : Override default cov_method.
                      'sample' | 'ledoit_wolf' | 'shrinkage' | 'oas' |
                      'minimum_covariance_determinant'.

        Returns:
            Covariance matrix (N, N).
        """
        m = method or self.cov_method
        return _estimate_covariance(returns, m)

    def portfolio_volatility(
        self,
        weights: np.ndarray | dict,
        returns: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> float:
        """
        Annualised portfolio volatility.

        Args:
            weights    : Asset weights (array or dict).
            returns    : Returns DataFrame (used if cov_matrix not provided).
            cov_matrix : Precomputed covariance matrix.

        Returns:
            Annualised volatility.
        """
        w, cov = self._resolve_w_cov(weights, returns, cov_matrix)
        return float(portfolio_volatility(w, cov * self.ann_factor))

    def marginal_risk_contribution(
        self,
        weights: np.ndarray | dict,
        returns: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Marginal risk contribution vector."""
        w, cov = self._resolve_w_cov(weights, returns, cov_matrix)
        return marginal_risk_contribution(w, cov * self.ann_factor)

    def component_risk_contribution(
        self,
        weights: np.ndarray | dict,
        returns: Optional[pd.DataFrame] = None,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Component (percentage) risk contributions."""
        w, cov = self._resolve_w_cov(weights, returns, cov_matrix)
        return component_risk_contribution(w, cov * self.ann_factor)

    def portfolio_var(
        self,
        weights: np.ndarray | dict,
        returns: pd.DataFrame,
        confidence: float = 0.99,
        method: str = "historical",
    ) -> float:
        """Portfolio Value at Risk."""
        if isinstance(weights, dict):
            assets = list(weights.keys())
            w = np.array([weights[a] for a in assets])
            R = returns[assets].values.astype(np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
            R = returns.values.astype(np.float64)
        return portfolio_var(w, R, confidence, method)

    def portfolio_cvar(
        self,
        weights: np.ndarray | dict,
        returns: pd.DataFrame,
        confidence: float = 0.99,
    ) -> float:
        """Portfolio Conditional VaR (Expected Shortfall)."""
        if isinstance(weights, dict):
            assets = list(weights.keys())
            w = np.array([weights[a] for a in assets])
            R = returns[assets].values.astype(np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
            R = returns.values.astype(np.float64)
        return portfolio_cvar(w, R, confidence)

    def portfolio_beta(
        self,
        portfolio_returns: np.ndarray | pd.Series,
        benchmark_returns: np.ndarray | pd.Series,
    ) -> float:
        """Portfolio beta vs. benchmark."""
        return portfolio_beta(
            np.asarray(portfolio_returns, dtype=np.float64),
            np.asarray(benchmark_returns, dtype=np.float64),
        )

    def tracking_error(
        self,
        portfolio_returns: np.ndarray | pd.Series,
        benchmark_returns: np.ndarray | pd.Series,
    ) -> float:
        """Annualised tracking error."""
        return tracking_error(
            np.asarray(portfolio_returns, dtype=np.float64),
            np.asarray(benchmark_returns, dtype=np.float64),
            self.ann_factor,
        )

    def information_ratio(
        self,
        portfolio_returns: np.ndarray | pd.Series,
        benchmark_returns: np.ndarray | pd.Series,
    ) -> float:
        """Annualised information ratio."""
        return information_ratio(
            np.asarray(portfolio_returns, dtype=np.float64),
            np.asarray(benchmark_returns, dtype=np.float64),
            self.ann_factor,
        )

    def stress_test_portfolio(
        self,
        weights: np.ndarray | dict,
        stress_scenarios: dict[str, np.ndarray],
        asset_names: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """Apply stress scenarios and return per-scenario P&L."""
        if isinstance(weights, dict):
            assets = list(weights.keys())
            w = np.array([weights[a] for a in assets])
        else:
            w = np.asarray(weights, dtype=np.float64)
        return stress_test_portfolio(w, stress_scenarios)

    def full_risk_report(
        self,
        weights: np.ndarray | dict,
        returns: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        asset_names: Optional[list[str]] = None,
    ) -> dict:
        """
        Compute a full risk report for a portfolio.

        Returns:
            Dict with keys:
                volatility, var_99, cvar_99, var_95, cvar_95,
                component_risk (array), beta, tracking_error, information_ratio,
                sharpe, sortino, calmar, max_drawdown.
        """
        if isinstance(weights, dict):
            assets = list(weights.keys())
            w = np.array([weights[a] for a in assets])
            R_df = returns[assets]
        else:
            w = np.asarray(weights, dtype=np.float64)
            R_df = returns
            assets = list(returns.columns)

        R = R_df.values.astype(np.float64)
        port_rets = R @ w

        cov = self.covariance_estimation(R_df)

        ann_vol = float(portfolio_volatility(w, cov * self.ann_factor))
        var99 = self.portfolio_var(w, R_df, 0.99)
        cvar99 = self.portfolio_cvar(w, R_df, 0.99)
        var95 = self.portfolio_var(w, R_df, 0.95)
        cvar95 = self.portfolio_cvar(w, R_df, 0.95)
        crc = component_risk_contribution(w, cov * self.ann_factor)

        # Performance metrics
        ann_ret = float(port_rets.mean() * self.ann_factor)
        sharpe = ann_ret / (ann_vol + 1e-12)

        downside = port_rets[port_rets < 0.0]
        sortino_denom = float(downside.std() * math.sqrt(self.ann_factor)) + 1e-12
        sortino = ann_ret / sortino_denom

        equity = np.cumprod(1.0 + port_rets)
        running_max = np.maximum.accumulate(equity)
        max_dd = float(np.max((running_max - equity) / (running_max + 1e-12)))
        calmar = ann_ret / (max_dd + 1e-12)

        report = {
            "volatility": ann_vol,
            "var_99": var99,
            "cvar_99": cvar99,
            "var_95": var95,
            "cvar_95": cvar95,
            "component_risk": dict(zip(assets, crc.tolist())),
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_dd,
            "annualised_return": ann_ret,
        }

        if benchmark_returns is not None:
            b = np.asarray(benchmark_returns, dtype=np.float64)
            min_len = min(len(port_rets), len(b))
            report["beta"] = portfolio_beta(port_rets[:min_len], b[:min_len])
            report["tracking_error"] = tracking_error(port_rets[:min_len], b[:min_len], self.ann_factor)
            report["information_ratio"] = information_ratio(port_rets[:min_len], b[:min_len], self.ann_factor)

        return report

    def plot_risk_contributions(
        self,
        weights: np.ndarray | dict,
        cov_matrix: np.ndarray,
        asset_names: Optional[list[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot component risk contributions as a pie chart.

        Args:
            weights     : Asset weights.
            cov_matrix  : Covariance matrix.
            asset_names : Labels for assets.
            save_path   : If provided, save figure to this path.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available.")
            return

        if isinstance(weights, dict):
            names = list(weights.keys())
            w = np.array([weights[a] for a in names])
        else:
            w = np.asarray(weights, dtype=np.float64)
            names = asset_names or [f"Asset {i+1}" for i in range(len(w))]

        crc = component_risk_contribution(w, cov_matrix)
        crc_pct = np.maximum(crc, 0.0)
        if crc_pct.sum() < 1e-12:
            crc_pct = np.ones(len(w)) / len(w)
        else:
            crc_pct /= crc_pct.sum()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Pie chart
        ax1.pie(
            crc_pct,
            labels=names,
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False,
        )
        ax1.set_title("Component Risk Contributions")

        # Bar chart: weights vs risk
        x = np.arange(len(names))
        width = 0.35
        ax2.bar(x - width / 2, w, width, label="Weight", color="steelblue", alpha=0.8)
        ax2.bar(x + width / 2, crc_pct, width, label="Risk Contribution", color="darkorange", alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha="right")
        ax2.set_title("Weights vs Risk Contributions")
        ax2.legend()
        ax2.set_ylabel("Fraction")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_w_cov(
        self,
        weights: np.ndarray | dict,
        returns: Optional[pd.DataFrame],
        cov_matrix: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(weights, dict):
            assets = list(weights.keys())
            w = np.array([weights[a] for a in assets], dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)

        if cov_matrix is not None:
            cov = np.asarray(cov_matrix, dtype=np.float64)
        elif returns is not None:
            cov = self.covariance_estimation(returns)
        else:
            raise ValueError("Either returns or cov_matrix must be provided.")

        return w, cov


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _skewness(x: np.ndarray) -> float:
    mu = x.mean()
    sigma = x.std() + 1e-12
    return float(np.mean(((x - mu) / sigma) ** 3))


def _excess_kurtosis(x: np.ndarray) -> float:
    mu = x.mean()
    sigma = x.std() + 1e-12
    return float(np.mean(((x - mu) / sigma) ** 4)) - 3.0


def compute_drawdown_series(returns: np.ndarray) -> np.ndarray:
    """
    Compute the drawdown series from a returns array.

    Args:
        returns : 1-D array of period returns.

    Returns:
        Drawdown series (positive values = drawdown).
    """
    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    return (running_max - equity) / (running_max + 1e-12)


def maximum_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from a returns array."""
    return float(np.max(compute_drawdown_series(returns)))


def calmar_ratio(returns: np.ndarray, ann_factor: int = 252) -> float:
    """Annualised return / Maximum drawdown."""
    ann_ret = float(np.mean(returns) * ann_factor)
    max_dd = maximum_drawdown(returns) + 1e-12
    return ann_ret / max_dd


def sortino_ratio(
    returns: np.ndarray,
    mar: float = 0.0,
    ann_factor: int = 252,
) -> float:
    """Annualised Sortino ratio."""
    excess = returns - mar
    downside = excess[excess < 0.0]
    if len(downside) == 0:
        return float(np.mean(excess) * ann_factor / 1e-12)
    std_d = float(np.std(downside) * math.sqrt(ann_factor)) + 1e-12
    return float(np.mean(excess) * ann_factor / std_d)
