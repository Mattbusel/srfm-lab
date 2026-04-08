"""
Advanced portfolio risk analytics.

Marginal/Component/Incremental VaR, Expected Shortfall decomposition,
stress testing, reverse stress tests, risk budgeting, factor risk,
tail risk (EVT), liquidity-adjusted VaR, correlation stress, drawdown risk,
Greeks risk, and credit risk.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List, Dict, Any


# ---------------------------------------------------------------------------
# 1. VaR Variants: Marginal, Component, Incremental
# ---------------------------------------------------------------------------

class VaRDecomposition:
    """Value-at-Risk decomposition: marginal, component, incremental."""

    def __init__(self, returns: np.ndarray, weights: np.ndarray,
                 confidence: float = 0.95):
        """
        returns: (T, N) asset returns
        weights: (N,) portfolio weights
        """
        self.returns = returns
        self.weights = weights
        self.confidence = confidence
        self.T, self.N = returns.shape
        self.cov = np.cov(returns, rowvar=False, bias=False)
        self.port_returns = returns @ weights
        self.port_vol = np.sqrt(weights @ self.cov @ weights)

    def parametric_var(self) -> float:
        from scipy.stats import norm
        z = norm.ppf(1 - self.confidence)
        mu = np.mean(self.port_returns)
        return -(mu + z * self.port_vol)

    def historical_var(self) -> float:
        return -float(np.percentile(self.port_returns, (1 - self.confidence) * 100))

    def marginal_var(self) -> np.ndarray:
        """dVaR/dw_i: sensitivity of portfolio VaR to weight changes."""
        from scipy.stats import norm
        z = norm.ppf(1 - self.confidence)
        return -z * (self.cov @ self.weights) / (self.port_vol + 1e-30)

    def component_var(self) -> np.ndarray:
        """Contribution of each asset to total VaR. Sums to total VaR."""
        mvar = self.marginal_var()
        return self.weights * mvar

    def incremental_var(self, delta_weights: np.ndarray) -> float:
        """Approximate VaR change from weight perturbation."""
        mvar = self.marginal_var()
        return float(np.sum(mvar * delta_weights))

    def pct_contribution(self) -> np.ndarray:
        cvar = self.component_var()
        total = np.sum(cvar)
        return cvar / (total + 1e-30) * 100


# ---------------------------------------------------------------------------
# 2. Expected Shortfall Decomposition
# ---------------------------------------------------------------------------

class ESDecomposition:
    """Expected Shortfall (CVaR) decomposition by position."""

    def __init__(self, returns: np.ndarray, weights: np.ndarray,
                 confidence: float = 0.95):
        self.returns = returns
        self.weights = weights
        self.confidence = confidence
        self.port_returns = returns @ weights

    def expected_shortfall(self) -> float:
        cutoff = np.percentile(self.port_returns, (1 - self.confidence) * 100)
        tail = self.port_returns[self.port_returns <= cutoff]
        return -float(np.mean(tail)) if len(tail) > 0 else 0.0

    def component_es(self) -> np.ndarray:
        """Each asset's contribution to portfolio ES."""
        cutoff = np.percentile(self.port_returns, (1 - self.confidence) * 100)
        mask = self.port_returns <= cutoff
        if mask.sum() == 0:
            return np.zeros(self.returns.shape[1])
        tail_returns = self.returns[mask]
        mean_tail = tail_returns.mean(axis=0)
        return -self.weights * mean_tail

    def marginal_es(self) -> np.ndarray:
        cutoff = np.percentile(self.port_returns, (1 - self.confidence) * 100)
        mask = self.port_returns <= cutoff
        if mask.sum() == 0:
            return np.zeros(self.returns.shape[1])
        return -self.returns[mask].mean(axis=0)


# ---------------------------------------------------------------------------
# 3. Stress Testing Framework
# ---------------------------------------------------------------------------

class StressTestFramework:
    """Historical and hypothetical stress testing."""

    def __init__(self, weights: np.ndarray, asset_names: Optional[List[str]] = None):
        self.weights = weights
        self.N = len(weights)
        self.asset_names = asset_names or [f"asset_{i}" for i in range(self.N)]

    def historical_scenario(self, scenario_returns: np.ndarray) -> Dict[str, float]:
        """Apply historical scenario returns to current portfolio."""
        port_return = float(self.weights @ scenario_returns)
        contributions = self.weights * scenario_returns
        return {
            "portfolio_return": port_return,
            "contributions": {self.asset_names[i]: float(contributions[i])
                              for i in range(self.N)},
            "worst_contributor": self.asset_names[int(np.argmin(contributions))],
        }

    def predefined_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Standard stress scenarios with typical market moves."""
        scenarios = {
            "2008_financial_crisis": np.full(self.N, -0.35) + np.random.default_rng(1).standard_normal(self.N) * 0.10,
            "2020_covid_crash": np.full(self.N, -0.30) + np.random.default_rng(2).standard_normal(self.N) * 0.08,
            "2022_rate_hike": np.full(self.N, -0.15) + np.random.default_rng(3).standard_normal(self.N) * 0.05,
            "equity_crash_20pct": np.full(self.N, -0.20),
            "rates_up_200bps": np.full(self.N, -0.05),
            "vol_spike_2x": np.full(self.N, -0.10) + np.random.default_rng(4).standard_normal(self.N) * 0.15,
        }
        results = {}
        for name, s in scenarios.items():
            results[name] = self.historical_scenario(s)
        return results

    def hypothetical_scenario(self, shocks: Dict[str, float]) -> Dict[str, float]:
        """Custom scenario: {asset_name: return_shock}."""
        scenario = np.zeros(self.N)
        for name, shock in shocks.items():
            if name in self.asset_names:
                idx = self.asset_names.index(name)
                scenario[idx] = shock
        return self.historical_scenario(scenario)

    def sensitivity_ladder(self, shock_range: np.ndarray = None) -> Dict[str, np.ndarray]:
        """P&L for uniform shocks from -20% to +20%."""
        if shock_range is None:
            shock_range = np.linspace(-0.20, 0.20, 41)
        pnls = np.array([float(self.weights.sum() * s) for s in shock_range])
        return {"shocks": shock_range, "pnls": pnls}


# ---------------------------------------------------------------------------
# 4. Reverse Stress Test
# ---------------------------------------------------------------------------

class ReverseStressTest:
    """Find scenarios causing a target portfolio loss."""

    def __init__(self, weights: np.ndarray, cov: np.ndarray):
        self.weights = weights
        self.cov = cov
        self.N = len(weights)

    def find_scenario(self, target_loss: float, method: str = "most_likely") -> np.ndarray:
        """Find the most likely scenario causing exactly target_loss."""
        if method == "most_likely":
            # Most likely scenario under multivariate normal: proportional to Sigma @ w
            direction = self.cov @ self.weights
            direction /= (np.linalg.norm(direction) + 1e-30)
            port_sensitivity = self.weights @ direction
            if abs(port_sensitivity) < 1e-12:
                return np.zeros(self.N)
            scale = -target_loss / port_sensitivity
            return direction * scale
        elif method == "max_correlation":
            # Maximize correlation stress impact
            L = np.linalg.cholesky(self.cov + np.eye(self.N) * 1e-8)
            z = L.T @ self.weights
            z /= (np.linalg.norm(z) + 1e-30)
            scenario = L @ z
            port_ret = self.weights @ scenario
            if abs(port_ret) < 1e-12:
                return np.zeros(self.N)
            return scenario * (-target_loss / port_ret)
        return np.zeros(self.N)

    def boundary_search(self, target_loss: float, n_samples: int = 10000,
                        rng: Optional[np.random.Generator] = None) -> List[np.ndarray]:
        """Find multiple scenarios near the loss boundary."""
        rng = rng or np.random.default_rng(42)
        L = np.linalg.cholesky(self.cov + np.eye(self.N) * 1e-8)
        scenarios = []
        for _ in range(n_samples):
            z = rng.standard_normal(self.N)
            s = L @ z
            port_ret = self.weights @ s
            if abs(port_ret + target_loss) < target_loss * 0.05:
                scenarios.append(s)
        return scenarios[:10]


# ---------------------------------------------------------------------------
# 5. Risk Budgeting
# ---------------------------------------------------------------------------

class RiskBudgeting:
    """Allocate and track risk budget utilization."""

    def __init__(self, cov: np.ndarray, risk_budgets: np.ndarray):
        """risk_budgets: target fractional risk contribution per asset (sums to 1)."""
        self.cov = cov
        self.budgets = risk_budgets / (risk_budgets.sum() + 1e-30)
        self.N = len(risk_budgets)

    def risk_budget_weights(self, max_iter: int = 500, tol: float = 1e-8) -> np.ndarray:
        """Find weights such that risk contributions match budgets."""
        w = self.budgets.copy()
        for _ in range(max_iter):
            sigma_w = self.cov @ w
            port_vol = np.sqrt(w @ sigma_w)
            mrc = sigma_w / (port_vol + 1e-30)
            rc = w * mrc
            total_rc = np.sum(rc)
            target_rc = self.budgets * total_rc
            # Newton-like update
            w_new = w * (target_rc / (rc + 1e-30)) ** 0.5
            w_new = w_new / np.sum(w_new)
            if np.max(np.abs(w_new - w)) < tol:
                w = w_new
                break
            w = w_new
        return w

    def utilization(self, weights: np.ndarray) -> Dict[str, Any]:
        sigma_w = self.cov @ weights
        port_vol = np.sqrt(weights @ sigma_w)
        mrc = sigma_w / (port_vol + 1e-30)
        rc = weights * mrc
        total_rc = np.sum(rc)
        pct_rc = rc / (total_rc + 1e-30)
        utilization = pct_rc / (self.budgets + 1e-30)
        return {
            "risk_contributions": pct_rc,
            "budgets": self.budgets,
            "utilization": utilization,
            "max_overbudget": float(np.max(utilization)),
            "breaches": np.where(utilization > 1.1)[0].tolist(),
        }


# ---------------------------------------------------------------------------
# 6. Factor Risk Decomposition
# ---------------------------------------------------------------------------

class FactorRiskDecomposition:
    """Decompose portfolio risk into systematic vs idiosyncratic."""

    def __init__(self, returns: np.ndarray, factor_returns: np.ndarray,
                 weights: np.ndarray):
        self.returns = returns
        self.factor_returns = factor_returns
        self.weights = weights
        T, N = returns.shape
        K = factor_returns.shape[1]
        # Estimate factor loadings via OLS per asset
        self.betas = np.zeros((N, K))
        self.idio_var = np.zeros(N)
        for i in range(N):
            b = np.linalg.lstsq(factor_returns, returns[:, i], rcond=None)[0]
            self.betas[i] = b
            resid = returns[:, i] - factor_returns @ b
            self.idio_var[i] = np.var(resid)
        self.factor_cov = np.cov(factor_returns, rowvar=False)

    def decompose(self) -> Dict[str, Any]:
        port_beta = self.weights @ self.betas
        systematic_var = port_beta @ self.factor_cov @ port_beta
        idio_var = self.weights ** 2 @ self.idio_var
        total_var = systematic_var + idio_var
        return {
            "total_vol": float(np.sqrt(total_var) * np.sqrt(252)),
            "systematic_vol": float(np.sqrt(systematic_var) * np.sqrt(252)),
            "idiosyncratic_vol": float(np.sqrt(idio_var) * np.sqrt(252)),
            "systematic_pct": float(systematic_var / (total_var + 1e-30) * 100),
            "portfolio_betas": port_beta.tolist(),
        }

    def factor_marginal_risk(self) -> np.ndarray:
        port_beta = self.weights @ self.betas
        return self.factor_cov @ port_beta


# ---------------------------------------------------------------------------
# 7. Tail Risk: EVT VaR, Hill Estimator, GPD
# ---------------------------------------------------------------------------

class TailRiskAnalyzer:
    """Extreme Value Theory for tail risk measurement."""

    def __init__(self, returns: np.ndarray):
        self.returns = returns
        self.losses = -returns

    def hill_estimator(self, k: Optional[int] = None) -> float:
        """Hill estimator for tail index. Higher = thinner tail."""
        sorted_losses = np.sort(self.losses)[::-1]
        if k is None:
            k = max(int(len(sorted_losses) * 0.05), 10)
        k = min(k, len(sorted_losses) - 1)
        top_k = sorted_losses[:k]
        threshold = sorted_losses[k]
        if threshold <= 0:
            return 1.0
        alpha = k / np.sum(np.log(top_k / threshold + 1e-30))
        return max(float(alpha), 0.1)

    def gpd_fit(self, threshold_quantile: float = 0.95) -> Dict[str, float]:
        """Fit Generalized Pareto Distribution to exceedances."""
        threshold = np.quantile(self.losses, threshold_quantile)
        exceedances = self.losses[self.losses > threshold] - threshold
        if len(exceedances) < 5:
            return {"xi": 0.0, "sigma": float(np.std(self.losses)), "threshold": float(threshold)}
        # Method of moments for GPD
        mean_exc = np.mean(exceedances)
        var_exc = np.var(exceedances)
        if mean_exc <= 0:
            return {"xi": 0.0, "sigma": 1.0, "threshold": float(threshold)}
        xi = 0.5 * (mean_exc ** 2 / var_exc - 1)
        sigma = mean_exc * (1 - xi) / 2 if abs(1 - xi) > 1e-10 else mean_exc
        sigma = max(sigma, 1e-10)
        return {"xi": float(xi), "sigma": float(sigma), "threshold": float(threshold)}

    def evt_var(self, confidence: float = 0.99) -> float:
        """VaR using GPD tail fit."""
        gpd = self.gpd_fit()
        xi, sigma, u = gpd["xi"], gpd["sigma"], gpd["threshold"]
        n = len(self.losses)
        n_u = np.sum(self.losses > u)
        p = 1 - confidence
        if abs(xi) < 1e-8:
            var = u + sigma * np.log(n_u / (n * p + 1e-30))
        else:
            var = u + sigma / xi * (((n_u / (n * p + 1e-30)) ** xi) - 1)
        return float(var)

    def tail_summary(self) -> Dict[str, float]:
        return {
            "hill_index": self.hill_estimator(),
            "gpd_xi": self.gpd_fit()["xi"],
            "var_99_evt": self.evt_var(0.99),
            "var_995_evt": self.evt_var(0.995),
            "var_999_evt": self.evt_var(0.999),
            "max_loss": float(np.max(self.losses)),
            "skewness": float(np.mean(((self.returns - self.returns.mean()) / (self.returns.std() + 1e-10)) ** 3)),
            "excess_kurtosis": float(np.mean(((self.returns - self.returns.mean()) / (self.returns.std() + 1e-10)) ** 4) - 3),
        }


# ---------------------------------------------------------------------------
# 8. Liquidity-Adjusted VaR
# ---------------------------------------------------------------------------

class LiquidityAdjustedVaR:
    """VaR with holding period and liquidity cost adjustment."""

    def __init__(self, returns: np.ndarray, weights: np.ndarray,
                 adv: np.ndarray, position_values: np.ndarray):
        """
        adv: (N,) average daily volume in dollars
        position_values: (N,) dollar value of each position
        """
        self.returns = returns
        self.weights = weights
        self.adv = adv
        self.position_values = position_values
        self.cov = np.cov(returns, rowvar=False)

    def holding_period_var(self, confidence: float = 0.99,
                           holding_days: Optional[np.ndarray] = None) -> float:
        """Scale VaR by sqrt(holding period) per asset."""
        from scipy.stats import norm
        z = norm.ppf(confidence)
        if holding_days is None:
            # Estimate holding period from liquidity: days to liquidate 10% participation
            participation = 0.10
            holding_days = self.position_values / (self.adv * participation + 1e-10)
            holding_days = np.maximum(holding_days, 1.0)
        # Adjusted covariance: scale each entry by sqrt(h_i * h_j)
        H = np.sqrt(np.outer(holding_days, holding_days))
        cov_adj = self.cov * H
        port_vol_adj = np.sqrt(self.weights @ cov_adj @ self.weights)
        return float(z * port_vol_adj)

    def liquidity_cost(self, sigma_spread: Optional[np.ndarray] = None) -> float:
        """Expected cost of liquidation: half-spread + market impact."""
        if sigma_spread is None:
            sigma_spread = 0.001 * np.ones(len(self.weights))
        half_spread_cost = np.sum(np.abs(self.position_values) * sigma_spread * 0.5)
        impact_cost = np.sum(0.1 * np.sqrt(np.abs(self.position_values) / (self.adv + 1e-10)) *
                             np.abs(self.position_values) * 0.01)
        return float(half_spread_cost + impact_cost)

    def total_lavar(self, confidence: float = 0.99) -> Dict[str, float]:
        var = self.holding_period_var(confidence)
        liq_cost = self.liquidity_cost()
        return {
            "holding_period_var": var,
            "liquidity_cost": liq_cost,
            "total_lavar": var + liq_cost,
        }


# ---------------------------------------------------------------------------
# 9. Correlation Stress
# ---------------------------------------------------------------------------

class CorrelationStress:
    """What if correlations spike to extreme levels?"""

    def __init__(self, returns: np.ndarray, weights: np.ndarray):
        self.returns = returns
        self.weights = weights
        self.cov = np.cov(returns, rowvar=False)
        self.vols = np.sqrt(np.diag(self.cov))
        self.corr = np.corrcoef(returns, rowvar=False)

    def stressed_var(self, target_corr: float = 0.8,
                     confidence: float = 0.99) -> Dict[str, float]:
        from scipy.stats import norm
        z = norm.ppf(confidence)
        N = len(self.weights)
        stressed_corr = np.full((N, N), target_corr)
        np.fill_diagonal(stressed_corr, 1.0)
        stressed_cov = np.outer(self.vols, self.vols) * stressed_corr
        port_vol_normal = np.sqrt(self.weights @ self.cov @ self.weights)
        port_vol_stressed = np.sqrt(self.weights @ stressed_cov @ self.weights)
        return {
            "normal_var": float(z * port_vol_normal),
            "stressed_var": float(z * port_vol_stressed),
            "var_increase_pct": float((port_vol_stressed / (port_vol_normal + 1e-30) - 1) * 100),
            "target_correlation": target_corr,
        }

    def correlation_sensitivity(self, corr_range: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        if corr_range is None:
            corr_range = np.linspace(0.0, 0.99, 50)
        vars_at_corr = []
        for c in corr_range:
            result = self.stressed_var(target_corr=c)
            vars_at_corr.append(result["stressed_var"])
        return {"correlations": corr_range, "vars": np.array(vars_at_corr)}

    def eigenvalue_stress(self, first_eigenvalue_mult: float = 2.0) -> Dict[str, float]:
        eigvals, eigvecs = np.linalg.eigh(self.cov)
        eigvals_stressed = eigvals.copy()
        eigvals_stressed[-1] *= first_eigenvalue_mult
        cov_stressed = eigvecs @ np.diag(eigvals_stressed) @ eigvecs.T
        from scipy.stats import norm
        z = norm.ppf(0.99)
        vol_normal = np.sqrt(self.weights @ self.cov @ self.weights)
        vol_stressed = np.sqrt(self.weights @ cov_stressed @ self.weights)
        return {
            "normal_vol": float(vol_normal * np.sqrt(252)),
            "stressed_vol": float(vol_stressed * np.sqrt(252)),
            "increase_pct": float((vol_stressed / (vol_normal + 1e-30) - 1) * 100),
        }


# ---------------------------------------------------------------------------
# 10. Drawdown Risk: Calmar, CDaR, Conditional Drawdown
# ---------------------------------------------------------------------------

class DrawdownRisk:
    """Drawdown-based risk measures."""

    def __init__(self, returns: np.ndarray):
        self.returns = returns
        self.cum = np.cumprod(1 + returns)
        self.running_max = np.maximum.accumulate(self.cum)
        self.drawdowns = (self.cum - self.running_max) / (self.running_max + 1e-10)

    def max_drawdown(self) -> float:
        return float(abs(np.min(self.drawdowns)))

    def calmar_ratio(self, periods: int = 252) -> float:
        ann_ret = np.mean(self.returns) * periods
        mdd = self.max_drawdown()
        return float(ann_ret / (mdd + 1e-10))

    def conditional_drawdown_at_risk(self, confidence: float = 0.95) -> float:
        """CDaR: average of worst (1-alpha) fraction of drawdowns."""
        sorted_dd = np.sort(self.drawdowns)
        cutoff_idx = int(len(sorted_dd) * (1 - confidence))
        cutoff_idx = max(cutoff_idx, 1)
        return float(abs(np.mean(sorted_dd[:cutoff_idx])))

    def ulcer_index(self) -> float:
        return float(np.sqrt(np.mean(self.drawdowns ** 2)))

    def drawdown_duration(self) -> Dict[str, Any]:
        durations = []
        in_dd = False
        start = 0
        for i in range(len(self.drawdowns)):
            if self.drawdowns[i] < -1e-6 and not in_dd:
                in_dd = True
                start = i
            elif self.drawdowns[i] >= -1e-6 and in_dd:
                in_dd = False
                durations.append(i - start)
        if in_dd:
            durations.append(len(self.drawdowns) - start)
        return {
            "max_duration": max(durations) if durations else 0,
            "avg_duration": float(np.mean(durations)) if durations else 0,
            "n_drawdowns": len(durations),
        }

    def summary(self) -> Dict[str, float]:
        return {
            "max_drawdown": self.max_drawdown(),
            "calmar": self.calmar_ratio(),
            "cdar_95": self.conditional_drawdown_at_risk(0.95),
            "ulcer_index": self.ulcer_index(),
            **self.drawdown_duration(),
        }


# ---------------------------------------------------------------------------
# 11. Greeks Risk: Delta/Gamma/Vega P&L Scenarios
# ---------------------------------------------------------------------------

class GreeksRisk:
    """Scenario analysis for options-like exposures via Greeks."""

    def __init__(self, deltas: np.ndarray, gammas: np.ndarray,
                 vegas: np.ndarray, thetas: np.ndarray,
                 notionals: np.ndarray):
        self.deltas = deltas
        self.gammas = gammas
        self.vegas = vegas
        self.thetas = thetas
        self.notionals = notionals
        self.N = len(deltas)

    def pnl_scenario(self, spot_moves: np.ndarray, vol_moves: np.ndarray,
                     dt: float = 1 / 252) -> Dict[str, Any]:
        delta_pnl = self.deltas * self.notionals * spot_moves
        gamma_pnl = 0.5 * self.gammas * self.notionals * spot_moves ** 2
        vega_pnl = self.vegas * self.notionals * vol_moves
        theta_pnl = self.thetas * self.notionals * dt
        total_per_asset = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
        return {
            "total_pnl": float(np.sum(total_per_asset)),
            "delta_pnl": float(np.sum(delta_pnl)),
            "gamma_pnl": float(np.sum(gamma_pnl)),
            "vega_pnl": float(np.sum(vega_pnl)),
            "theta_pnl": float(np.sum(theta_pnl)),
            "per_asset": total_per_asset,
        }

    def scenario_grid(self, spot_range: np.ndarray = None,
                      vol_range: np.ndarray = None) -> Dict[str, np.ndarray]:
        if spot_range is None:
            spot_range = np.linspace(-0.10, 0.10, 21)
        if vol_range is None:
            vol_range = np.linspace(-0.10, 0.10, 21)
        pnl_grid = np.zeros((len(spot_range), len(vol_range)))
        for i, ds in enumerate(spot_range):
            for j, dv in enumerate(vol_range):
                s_moves = np.full(self.N, ds)
                v_moves = np.full(self.N, dv)
                result = self.pnl_scenario(s_moves, v_moves)
                pnl_grid[i, j] = result["total_pnl"]
        return {"spot_range": spot_range, "vol_range": vol_range, "pnl_grid": pnl_grid}

    def worst_case(self, spot_range: Tuple[float, float] = (-0.15, 0.15),
                   vol_range: Tuple[float, float] = (-0.20, 0.20),
                   n_samples: int = 5000) -> Dict[str, Any]:
        rng = np.random.default_rng(42)
        worst_pnl = np.inf
        worst_scenario = None
        for _ in range(n_samples):
            ds = rng.uniform(spot_range[0], spot_range[1], self.N)
            dv = rng.uniform(vol_range[0], vol_range[1], self.N)
            result = self.pnl_scenario(ds, dv)
            if result["total_pnl"] < worst_pnl:
                worst_pnl = result["total_pnl"]
                worst_scenario = {"spot_moves": ds.copy(), "vol_moves": dv.copy()}
        return {"worst_pnl": float(worst_pnl), "worst_scenario": worst_scenario}


# ---------------------------------------------------------------------------
# 12. Credit Risk: Default Probability Weighted Loss
# ---------------------------------------------------------------------------

class CreditRisk:
    """Credit risk analytics: PD-weighted loss, CVA, concentration."""

    def __init__(self, exposures: np.ndarray, default_probs: np.ndarray,
                 recovery_rates: np.ndarray, correlations: Optional[np.ndarray] = None):
        self.exposures = exposures
        self.pds = default_probs
        self.rrs = recovery_rates
        self.N = len(exposures)
        self.lgds = 1 - recovery_rates
        if correlations is not None:
            self.corr = correlations
        else:
            self.corr = np.eye(self.N) * 0.3 + np.ones((self.N, self.N)) * 0.7
            np.fill_diagonal(self.corr, 1.0)

    def expected_loss(self) -> float:
        return float(np.sum(self.exposures * self.pds * self.lgds))

    def unexpected_loss(self) -> float:
        var_per = self.exposures ** 2 * self.lgds ** 2 * self.pds * (1 - self.pds)
        cov_matrix = np.outer(np.sqrt(var_per), np.sqrt(var_per)) * self.corr
        return float(np.sqrt(np.sum(cov_matrix)))

    def credit_var(self, confidence: float = 0.99, n_simulations: int = 50000,
                   rng: Optional[np.random.Generator] = None) -> float:
        rng = rng or np.random.default_rng(42)
        try:
            L = np.linalg.cholesky(self.corr + np.eye(self.N) * 1e-6)
        except np.linalg.LinAlgError:
            L = np.eye(self.N)
        losses = np.zeros(n_simulations)
        for s in range(n_simulations):
            z = L @ rng.standard_normal(self.N)
            from scipy.stats import norm
            thresholds = norm.ppf(self.pds)
            defaults = z < thresholds
            losses[s] = np.sum(self.exposures[defaults] * self.lgds[defaults])
        return float(np.percentile(losses, confidence * 100))

    def concentration_index(self) -> float:
        """Herfindahl index of exposure concentration."""
        shares = self.exposures / (np.sum(self.exposures) + 1e-30)
        return float(np.sum(shares ** 2))

    def summary(self) -> Dict[str, float]:
        return {
            "expected_loss": self.expected_loss(),
            "unexpected_loss": self.unexpected_loss(),
            "total_exposure": float(np.sum(self.exposures)),
            "avg_pd": float(np.mean(self.pds)),
            "avg_lgd": float(np.mean(self.lgds)),
            "concentration_hhi": self.concentration_index(),
        }


# ---------------------------------------------------------------------------
# Comprehensive Risk Report
# ---------------------------------------------------------------------------

def comprehensive_risk_report(returns: np.ndarray, weights: np.ndarray,
                               factor_returns: Optional[np.ndarray] = None,
                               adv: Optional[np.ndarray] = None,
                               asset_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Generate full risk analytics report."""
    T, N = returns.shape
    report: Dict[str, Any] = {}

    # VaR decomposition
    vd = VaRDecomposition(returns, weights)
    report["var"] = {
        "parametric_var": vd.parametric_var(),
        "historical_var": vd.historical_var(),
        "component_var": vd.component_var().tolist(),
        "pct_contribution": vd.pct_contribution().tolist(),
    }

    # ES
    es = ESDecomposition(returns, weights)
    report["expected_shortfall"] = {
        "es_95": es.expected_shortfall(),
        "component_es": es.component_es().tolist(),
    }

    # Stress tests
    st = StressTestFramework(weights, asset_names)
    report["stress_tests"] = st.predefined_scenarios()

    # Tail risk
    port_returns = returns @ weights
    tr = TailRiskAnalyzer(port_returns)
    report["tail_risk"] = tr.tail_summary()

    # Drawdown
    dd = DrawdownRisk(port_returns)
    report["drawdown"] = dd.summary()

    # Correlation stress
    cs = CorrelationStress(returns, weights)
    report["correlation_stress"] = cs.stressed_var(0.8)

    # Factor decomposition
    if factor_returns is not None:
        fd = FactorRiskDecomposition(returns, factor_returns, weights)
        report["factor_risk"] = fd.decompose()

    # Liquidity
    if adv is not None:
        position_values = np.abs(weights) * 1e6
        lv = LiquidityAdjustedVaR(returns, weights, adv, position_values)
        report["liquidity_var"] = lv.total_lavar()

    return report
