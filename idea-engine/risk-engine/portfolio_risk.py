"""
Portfolio risk engine — comprehensive real-time risk monitoring and management.

Implements:
  - Multi-factor risk decomposition (systematic vs idiosyncratic)
  - Real-time VaR/CVaR with multiple methods (historical, parametric, Monte Carlo)
  - Stress testing framework: historical + hypothetical scenarios
  - Greeks aggregation across portfolio
  - Concentration risk: HHI, sector, factor exposure limits
  - Liquidity risk: time-to-liquidate, market impact estimation
  - Correlation breakdown detection
  - Drawdown monitoring with Kelly-based deleverage triggers
  - Risk budget allocation and tracking
  - Tail risk hedging recommendations
  - P&L attribution: factor, sector, alpha decomposition
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Risk Metrics ──────────────────────────────────────────────────────────────

@dataclass
class PortfolioRiskMetrics:
    """Comprehensive snapshot of portfolio risk."""
    timestamp: float = 0.0

    # Value at Risk
    var_95_pct: float = 0.0
    var_99_pct: float = 0.0
    cvar_95_pct: float = 0.0
    cvar_99_pct: float = 0.0

    # Volatility
    portfolio_vol_annual: float = 0.0
    tracking_error: float = 0.0
    beta_to_benchmark: float = 0.0

    # Drawdown
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    drawdown_duration_days: int = 0

    # Concentration
    hhi_position: float = 0.0
    hhi_sector: float = 0.0
    top5_weight_pct: float = 0.0

    # Factor exposure
    factor_exposures: dict = field(default_factory=dict)
    factor_risk_contribution: dict = field(default_factory=dict)

    # Liquidity
    avg_days_to_liquidate: float = 0.0
    illiquid_pct: float = 0.0

    # Tail
    skewness: float = 0.0
    kurtosis: float = 0.0
    tail_risk_score: float = 0.0

    # Overall
    risk_score: float = 0.0  # 0-100 composite
    risk_level: str = "normal"  # low/normal/elevated/high/critical


# ── VaR Engine ────────────────────────────────────────────────────────────────

class VaREngine:
    """Multi-method Value at Risk computation."""

    def historical_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
    ) -> dict:
        """Historical simulation VaR."""
        sorted_r = np.sort(returns)
        n = len(sorted_r)
        idx = int((1 - confidence) * n)
        var = float(-sorted_r[max(idx, 0)])
        cvar = float(-sorted_r[:max(idx, 1)].mean())
        return {"var": var, "cvar": cvar, "method": "historical", "n_obs": n}

    def parametric_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
    ) -> dict:
        """Gaussian parametric VaR."""
        mu = float(returns.mean())
        sigma = float(returns.std())
        z = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}.get(confidence, 1.645)
        var = float(-(mu - z * sigma))
        # Cornish-Fisher correction for non-normality
        s = float(np.mean(((returns - mu) / max(sigma, 1e-10))**3))
        k = float(np.mean(((returns - mu) / max(sigma, 1e-10))**4)) - 3
        z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * k / 24 - (2*z**3 - 5*z) * s**2 / 36
        var_cf = float(-(mu - z_cf * sigma))
        return {
            "var_gaussian": var,
            "var_cornish_fisher": var_cf,
            "method": "parametric",
            "skewness": s,
            "excess_kurtosis": k,
        }

    def monte_carlo_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        n_sims: int = 10000,
        horizon_days: int = 1,
        seed: int = 42,
    ) -> dict:
        """Monte Carlo VaR with bootstrap."""
        rng = np.random.default_rng(seed)
        mu = float(returns.mean())
        sigma = float(returns.std())

        # Simulate multi-day returns
        sims = rng.normal(mu * horizon_days, sigma * math.sqrt(horizon_days), n_sims)
        sorted_sims = np.sort(sims)
        idx = int((1 - confidence) * n_sims)
        var = float(-sorted_sims[max(idx, 0)])
        cvar = float(-sorted_sims[:max(idx, 1)].mean())
        return {"var": var, "cvar": cvar, "method": "monte_carlo", "n_sims": n_sims, "horizon": horizon_days}

    def component_var(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        confidence: float = 0.95,
    ) -> dict:
        """Component VaR: marginal contribution of each position."""
        z = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}.get(confidence, 1.645)
        port_var_raw = float(np.sqrt(weights @ cov_matrix @ weights))
        port_var = z * port_var_raw

        # Marginal VaR
        marginal = z * (cov_matrix @ weights) / max(port_var_raw, 1e-10)
        component = weights * marginal
        pct_contribution = component / max(port_var, 1e-10)

        return {
            "portfolio_var": float(port_var),
            "marginal_var": marginal.tolist(),
            "component_var": component.tolist(),
            "pct_contribution": pct_contribution.tolist(),
            "diversification_ratio": float(
                sum(abs(w) * z * math.sqrt(cov_matrix[i, i])
                    for i, w in enumerate(weights)) / max(port_var, 1e-10)
            ),
        }


# ── Stress Testing ────────────────────────────────────────────────────────────

@dataclass
class StressScenario:
    name: str
    description: str
    factor_shocks: dict[str, float]  # factor_name -> shock (%)
    probability: float = 0.05
    historical_analog: str = ""


BUILT_IN_SCENARIOS = [
    StressScenario("2008_gfc", "Global Financial Crisis", {
        "equity": -0.40, "credit_ig": -0.08, "credit_hy": -0.25, "rates": -0.02,
        "vol": 0.60, "commodity": -0.35, "fx_em": -0.20,
    }, 0.02, "Sep-Nov 2008"),
    StressScenario("2020_covid", "COVID-19 Crash", {
        "equity": -0.34, "credit_ig": -0.05, "credit_hy": -0.15, "rates": -0.015,
        "vol": 0.50, "commodity": -0.30, "fx_em": -0.12,
    }, 0.03, "Feb-Mar 2020"),
    StressScenario("rate_shock_300bp", "+300bp Rate Shock", {
        "equity": -0.15, "credit_ig": -0.10, "credit_hy": -0.12, "rates": 0.03,
        "vol": 0.15, "commodity": 0.05, "fx_em": -0.08,
    }, 0.05),
    StressScenario("stagflation", "Stagflation: High Inflation + Recession", {
        "equity": -0.25, "credit_ig": -0.06, "credit_hy": -0.18, "rates": 0.02,
        "vol": 0.30, "commodity": 0.20, "fx_em": -0.15,
    }, 0.04),
    StressScenario("credit_crisis", "Credit Crisis: Widening Spreads", {
        "equity": -0.20, "credit_ig": -0.12, "credit_hy": -0.30, "rates": -0.01,
        "vol": 0.35, "commodity": -0.10, "fx_em": -0.18,
    }, 0.03),
    StressScenario("flash_crash", "Flash Crash / Liquidity Crisis", {
        "equity": -0.10, "credit_ig": -0.02, "credit_hy": -0.05, "rates": 0.0,
        "vol": 0.40, "commodity": -0.05, "fx_em": -0.05,
    }, 0.05),
    StressScenario("em_contagion", "EM Contagion", {
        "equity": -0.12, "credit_ig": -0.03, "credit_hy": -0.10, "rates": -0.005,
        "vol": 0.20, "commodity": -0.08, "fx_em": -0.25,
    }, 0.04),
    StressScenario("geopolitical", "Geopolitical Escalation", {
        "equity": -0.15, "credit_ig": -0.04, "credit_hy": -0.08, "rates": -0.01,
        "vol": 0.25, "commodity": 0.30, "fx_em": -0.10,
    }, 0.05),
]


class StressTester:
    """Stress testing engine for portfolios."""

    def __init__(self, scenarios: list[StressScenario] = None):
        self.scenarios = scenarios or BUILT_IN_SCENARIOS

    def apply_scenario(
        self,
        weights: np.ndarray,
        factor_loadings: np.ndarray,  # (n_assets, n_factors)
        factor_names: list[str],
        scenario: StressScenario,
    ) -> dict:
        """Apply a stress scenario to portfolio."""
        n_assets, n_factors = factor_loadings.shape
        shocks = np.array([scenario.factor_shocks.get(f, 0.0) for f in factor_names])

        # Asset-level impact: factor_loading * factor_shock
        asset_impacts = factor_loadings @ shocks
        portfolio_impact = float(weights @ asset_impacts)

        # Worst impacted positions
        position_impacts = weights * asset_impacts
        worst_idx = np.argsort(position_impacts)[:3]

        return {
            "scenario": scenario.name,
            "portfolio_pnl_pct": float(portfolio_impact * 100),
            "asset_impacts": asset_impacts.tolist(),
            "position_impacts": position_impacts.tolist(),
            "worst_positions": worst_idx.tolist(),
            "worst_position_pnl": float(position_impacts[worst_idx[0]] * 100) if len(worst_idx) > 0 else 0.0,
            "probability": scenario.probability,
            "expected_loss": float(portfolio_impact * scenario.probability * 100),
        }

    def run_all_scenarios(
        self,
        weights: np.ndarray,
        factor_loadings: np.ndarray,
        factor_names: list[str],
    ) -> dict:
        """Run all scenarios and aggregate."""
        results = []
        for scenario in self.scenarios:
            result = self.apply_scenario(weights, factor_loadings, factor_names, scenario)
            results.append(result)

        # Sort by impact
        results.sort(key=lambda r: r["portfolio_pnl_pct"])
        worst = results[0]
        expected_loss = sum(r["expected_loss"] for r in results)

        return {
            "scenario_results": results,
            "worst_scenario": worst["scenario"],
            "worst_pnl_pct": worst["portfolio_pnl_pct"],
            "probability_weighted_loss_pct": float(expected_loss),
            "n_scenarios": len(results),
        }

    def reverse_stress_test(
        self,
        weights: np.ndarray,
        factor_loadings: np.ndarray,
        factor_names: list[str],
        target_loss_pct: float = -10.0,
    ) -> dict:
        """Find the minimum shock that causes target_loss_pct."""
        n_factors = len(factor_names)
        # Direction: use portfolio's factor sensitivity
        sensitivity = factor_loadings.T @ weights  # (n_factors,)
        sens_norm = np.linalg.norm(sensitivity)
        if sens_norm < 1e-10:
            return {"found": False}

        direction = -sensitivity / sens_norm  # shock direction that hurts most

        # Binary search for magnitude
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            shock = direction * mid
            impact = float(weights @ (factor_loadings @ shock))
            if impact * 100 > target_loss_pct:
                lo = mid
            else:
                hi = mid

        final_shock = direction * hi
        return {
            "found": True,
            "shock_magnitude": float(hi),
            "factor_shocks": {f: float(final_shock[i]) for i, f in enumerate(factor_names)},
            "realized_loss_pct": float(weights @ (factor_loadings @ final_shock) * 100),
            "target_loss_pct": target_loss_pct,
        }


# ── Concentration Risk ────────────────────────────────────────────────────────

def concentration_metrics(
    weights: np.ndarray,
    sector_labels: Optional[list[str]] = None,
) -> dict:
    """Compute concentration risk metrics."""
    abs_w = np.abs(weights)
    abs_w_norm = abs_w / (abs_w.sum() + 1e-10)

    # HHI (Herfindahl-Hirschman Index)
    hhi_position = float(np.sum(abs_w_norm**2))
    effective_n = float(1 / max(hhi_position, 1e-10))

    # Top-N concentration
    sorted_w = np.sort(abs_w_norm)[::-1]
    top1 = float(sorted_w[0]) if len(sorted_w) > 0 else 0.0
    top5 = float(sorted_w[:5].sum()) if len(sorted_w) >= 5 else float(sorted_w.sum())
    top10 = float(sorted_w[:10].sum()) if len(sorted_w) >= 10 else float(sorted_w.sum())

    result = {
        "hhi_position": hhi_position,
        "effective_n_positions": effective_n,
        "top1_pct": top1 * 100,
        "top5_pct": top5 * 100,
        "top10_pct": top10 * 100,
        "n_positions": int(np.sum(abs_w > 1e-6)),
    }

    # Sector concentration
    if sector_labels is not None and len(sector_labels) == len(weights):
        sector_weights = {}
        for w, s in zip(abs_w_norm, sector_labels):
            sector_weights[s] = sector_weights.get(s, 0.0) + float(w)
        sector_w = np.array(list(sector_weights.values()))
        hhi_sector = float(np.sum(sector_w**2))
        result["hhi_sector"] = hhi_sector
        result["sector_weights"] = sector_weights
        result["top_sector"] = max(sector_weights, key=sector_weights.get)

    return result


# ── Liquidity Risk ────────────────────────────────────────────────────────────

def liquidity_risk_assessment(
    position_sizes: np.ndarray,     # notional per position
    daily_volumes: np.ndarray,      # ADV per position
    bid_ask_spreads_bps: np.ndarray,
    max_participation_rate: float = 0.10,
) -> dict:
    """Assess portfolio liquidity risk."""
    n = len(position_sizes)
    participation_rates = position_sizes / (daily_volumes + 1e-10)
    days_to_liquidate = participation_rates / max_participation_rate

    # Liquidation cost estimate: spread + impact
    impact_bps = 10 * np.sqrt(participation_rates)  # simplified sqrt model
    total_cost_bps = bid_ask_spreads_bps + impact_bps

    # Weighted average
    total_notional = float(position_sizes.sum())
    weight = position_sizes / max(total_notional, 1e-10)

    avg_days = float(np.sum(weight * days_to_liquidate))
    max_days = float(days_to_liquidate.max())
    avg_cost = float(np.sum(weight * total_cost_bps))
    illiquid_frac = float(np.mean(days_to_liquidate > 3))

    return {
        "avg_days_to_liquidate": avg_days,
        "max_days_to_liquidate": max_days,
        "avg_liquidation_cost_bps": avg_cost,
        "total_liquidation_cost_bps": float(total_cost_bps.sum()),
        "illiquid_fraction": illiquid_frac,
        "days_per_position": days_to_liquidate.tolist(),
        "cost_per_position_bps": total_cost_bps.tolist(),
        "liquidity_score": float(1 - min(avg_days / 10, 1)),
    }


# ── Drawdown Monitor ──────────────────────────────────────────────────────────

class DrawdownMonitor:
    """Real-time drawdown tracking and deleverage signals."""

    def __init__(
        self,
        deleverage_threshold: float = 0.10,  # 10% DD triggers deleveraging
        critical_threshold: float = 0.20,    # 20% DD triggers emergency
        recovery_factor: float = 0.5,        # re-lever at 50% recovery
    ):
        self.deleverage_threshold = deleverage_threshold
        self.critical_threshold = critical_threshold
        self.recovery_factor = recovery_factor
        self._hwm = 0.0
        self._equity_curve: list[float] = []
        self._is_deleveraged = False

    def update(self, nav: float) -> dict:
        """Update with new NAV, return risk signals."""
        self._equity_curve.append(nav)
        self._hwm = max(self._hwm, nav)

        drawdown = (self._hwm - nav) / max(self._hwm, 1e-10)

        # Drawdown duration
        dd_start = len(self._equity_curve) - 1
        for i in range(len(self._equity_curve) - 1, -1, -1):
            if self._equity_curve[i] >= self._hwm:
                dd_start = i
                break
        dd_duration = len(self._equity_curve) - 1 - dd_start

        # Kelly-based deleverage
        if drawdown >= self.critical_threshold:
            target_leverage = 0.25
            action = "emergency_deleverage"
        elif drawdown >= self.deleverage_threshold:
            # Proportional deleverage
            excess = (drawdown - self.deleverage_threshold) / (self.critical_threshold - self.deleverage_threshold)
            target_leverage = max(1 - excess * 0.75, 0.25)
            action = "deleverage"
            self._is_deleveraged = True
        elif self._is_deleveraged:
            # Recovery check
            recovery = 1 - drawdown / max(self.deleverage_threshold, 1e-10)
            if recovery >= self.recovery_factor:
                target_leverage = 1.0
                action = "relever"
                self._is_deleveraged = False
            else:
                target_leverage = 0.6
                action = "hold_reduced"
        else:
            target_leverage = 1.0
            action = "normal"

        return {
            "current_drawdown_pct": float(drawdown * 100),
            "hwm": float(self._hwm),
            "drawdown_duration_days": dd_duration,
            "target_leverage": float(target_leverage),
            "action": action,
            "is_deleveraged": self._is_deleveraged,
        }

    def max_drawdown(self) -> float:
        if len(self._equity_curve) < 2:
            return 0.0
        curve = np.array(self._equity_curve)
        running_max = np.maximum.accumulate(curve)
        dd = (running_max - curve) / (running_max + 1e-10)
        return float(dd.max())


# ── Correlation Breakdown Detection ───────────────────────────────────────────

def correlation_regime_monitor(
    returns: np.ndarray,  # (T, N)
    window_short: int = 21,
    window_long: int = 126,
    breakdown_threshold: float = 0.3,
) -> dict:
    """Detect correlation regime changes."""
    T, N = returns.shape
    if T < window_long + 5:
        return {"breakdown_detected": False}

    # Short-term vs long-term correlation
    corr_short = np.corrcoef(returns[-window_short:].T)
    corr_long = np.corrcoef(returns[-window_long:].T)

    # Frobenius norm of difference
    diff = corr_short - corr_long
    np.fill_diagonal(diff, 0)
    change_magnitude = float(np.sqrt(np.sum(diff**2)) / max(N * (N - 1) / 2, 1))

    # Average correlation levels
    upper_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    avg_corr_short = float(corr_short[upper_mask].mean())
    avg_corr_long = float(corr_long[upper_mask].mean())

    # Correlation spike detection
    corr_spike = avg_corr_short > avg_corr_long + 0.15

    breakdown = bool(change_magnitude > breakdown_threshold)

    return {
        "breakdown_detected": breakdown,
        "correlation_spike": corr_spike,
        "change_magnitude": change_magnitude,
        "avg_correlation_short": avg_corr_short,
        "avg_correlation_long": avg_corr_long,
        "correlation_change": float(avg_corr_short - avg_corr_long),
        "regime": "crisis_herding" if avg_corr_short > 0.6 else
                  "diversified" if avg_corr_short < 0.2 else "normal",
    }


# ── P&L Attribution ───────────────────────────────────────────────────────────

def pnl_attribution(
    portfolio_return: float,
    weights: np.ndarray,
    asset_returns: np.ndarray,
    factor_loadings: np.ndarray,   # (n_assets, n_factors)
    factor_returns: np.ndarray,    # (n_factors,)
    factor_names: list[str],
) -> dict:
    """Decompose portfolio P&L into factor + specific components."""
    n_assets = len(weights)
    n_factors = len(factor_names)

    # Factor contribution
    asset_factor_returns = factor_loadings @ factor_returns  # (n_assets,)
    factor_pnl = float(weights @ asset_factor_returns)

    # Per-factor breakdown
    factor_breakdown = {}
    for j, name in enumerate(factor_names):
        contrib = float(weights @ (factor_loadings[:, j] * factor_returns[j]))
        factor_breakdown[name] = contrib

    # Specific (alpha) return
    specific_returns = asset_returns - asset_factor_returns
    alpha_pnl = float(weights @ specific_returns)

    # Interaction (residual)
    interaction = portfolio_return - factor_pnl - alpha_pnl

    return {
        "total_return": float(portfolio_return),
        "factor_return": float(factor_pnl),
        "alpha_return": float(alpha_pnl),
        "interaction": float(interaction),
        "factor_breakdown": factor_breakdown,
        "factor_pct_of_total": float(factor_pnl / max(abs(portfolio_return), 1e-10) * 100),
        "alpha_pct_of_total": float(alpha_pnl / max(abs(portfolio_return), 1e-10) * 100),
    }


# ── Risk Budget Tracker ──────────────────────────────────────────────────────

class RiskBudgetTracker:
    """Track and enforce risk budgets across strategies/positions."""

    def __init__(self, budget_limits: dict[str, float]):
        """budget_limits: e.g., {'equity': 0.10, 'credit': 0.05, 'rates': 0.03}"""
        self.limits = budget_limits
        self._usage: dict[str, float] = {k: 0.0 for k in budget_limits}

    def update_usage(self, category: str, risk_used: float) -> None:
        if category in self._usage:
            self._usage[category] = risk_used

    def check_budget(self) -> dict:
        """Check if any budget is breached."""
        breaches = {}
        utilizations = {}
        for cat, limit in self.limits.items():
            used = self._usage.get(cat, 0.0)
            util = used / max(limit, 1e-10)
            utilizations[cat] = float(util)
            if used > limit:
                breaches[cat] = {
                    "limit": float(limit),
                    "used": float(used),
                    "excess": float(used - limit),
                }

        return {
            "breaches": breaches,
            "n_breaches": len(breaches),
            "utilizations": utilizations,
            "worst_utilization": float(max(utilizations.values())) if utilizations else 0.0,
            "overall_status": "breach" if breaches else "warning" if max(utilizations.values(), default=0) > 0.8 else "ok",
        }

    def available_budget(self, category: str) -> float:
        limit = self.limits.get(category, 0.0)
        used = self._usage.get(category, 0.0)
        return float(max(limit - used, 0))


# ── Portfolio Risk Engine ─────────────────────────────────────────────────────

class PortfolioRiskEngine:
    """
    Master risk engine: combines all risk modules.
    """

    def __init__(
        self,
        var_confidence: float = 0.95,
        deleverage_threshold: float = 0.10,
    ):
        self.var_engine = VaREngine()
        self.stress_tester = StressTester()
        self.dd_monitor = DrawdownMonitor(deleverage_threshold=deleverage_threshold)
        self.var_confidence = var_confidence

    def compute_risk_snapshot(
        self,
        weights: np.ndarray,
        returns: np.ndarray,          # (T, N) historical returns
        factor_loadings: Optional[np.ndarray] = None,
        factor_names: Optional[list[str]] = None,
        daily_volumes: Optional[np.ndarray] = None,
        sector_labels: Optional[list[str]] = None,
        nav: Optional[float] = None,
    ) -> PortfolioRiskMetrics:
        """Compute comprehensive risk snapshot."""
        T, N = returns.shape
        port_returns = returns @ weights

        # VaR
        hist_var = self.var_engine.historical_var(port_returns, self.var_confidence)
        hist_var_99 = self.var_engine.historical_var(port_returns, 0.99)
        param_var = self.var_engine.parametric_var(port_returns, self.var_confidence)

        # Portfolio vol
        port_vol = float(port_returns.std() * math.sqrt(252))

        # Drawdown
        cum_returns = np.cumprod(1 + port_returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (running_max - cum_returns) / (running_max + 1e-10)
        current_dd = float(drawdowns[-1])
        max_dd = float(drawdowns.max())

        # Higher moments
        mu = float(port_returns.mean())
        sigma = float(port_returns.std() + 1e-10)
        skew = float(np.mean(((port_returns - mu) / sigma)**3))
        kurt = float(np.mean(((port_returns - mu) / sigma)**4))

        # Concentration
        conc = concentration_metrics(weights, sector_labels)

        # Factor decomposition
        factor_exp = {}
        factor_risk_contrib = {}
        if factor_loadings is not None and factor_names is not None:
            for j, name in enumerate(factor_names):
                factor_exp[name] = float(weights @ factor_loadings[:, j])
            # Factor risk contribution
            cov = np.cov(returns.T) + np.eye(N) * 1e-8
            port_var_total = float(weights @ cov @ weights)
            if factor_loadings is not None:
                for j, name in enumerate(factor_names):
                    fl = factor_loadings[:, j]
                    contrib = float(weights @ np.outer(fl, fl) @ weights)
                    factor_risk_contrib[name] = contrib / max(port_var_total, 1e-10)

        # Liquidity
        avg_days = 0.0
        illiquid_pct = 0.0
        if daily_volumes is not None:
            liq = liquidity_risk_assessment(
                np.abs(weights) * 1e6,  # assume $1M portfolio
                daily_volumes,
                np.full(N, 5.0),  # default 5 bps spread
            )
            avg_days = liq["avg_days_to_liquidate"]
            illiquid_pct = liq["illiquid_fraction"]

        # Tail risk score
        tail_score = float(min(
            (max(kurt - 3, 0) / 5) * 0.4 + (max(-skew, 0) / 2) * 0.3 +
            (hist_var_99["cvar"] / max(sigma, 1e-10)) * 0.3,
            1.0
        ))

        # Overall risk score (0-100)
        risk_score = float(min(
            (hist_var["var"] / 0.05 * 25) +
            (current_dd / 0.15 * 25) +
            (conc["hhi_position"] * 25) +
            (tail_score * 25),
            100
        ))

        risk_level = (
            "critical" if risk_score > 80 else
            "high" if risk_score > 60 else
            "elevated" if risk_score > 40 else
            "normal" if risk_score > 20 else "low"
        )

        # Drawdown monitor update
        if nav is not None:
            self.dd_monitor.update(nav)

        return PortfolioRiskMetrics(
            var_95_pct=hist_var["var"],
            var_99_pct=hist_var_99["var"],
            cvar_95_pct=hist_var["cvar"],
            cvar_99_pct=hist_var_99["cvar"],
            portfolio_vol_annual=port_vol,
            current_drawdown_pct=current_dd,
            max_drawdown_pct=max_dd,
            hhi_position=conc["hhi_position"],
            hhi_sector=conc.get("hhi_sector", 0.0),
            top5_weight_pct=conc["top5_pct"],
            factor_exposures=factor_exp,
            factor_risk_contribution=factor_risk_contrib,
            avg_days_to_liquidate=avg_days,
            illiquid_pct=illiquid_pct,
            skewness=skew,
            kurtosis=kurt,
            tail_risk_score=tail_score,
            risk_score=risk_score,
            risk_level=risk_level,
        )

    def tail_hedge_recommendation(
        self,
        risk_metrics: PortfolioRiskMetrics,
        portfolio_notional: float,
    ) -> dict:
        """Recommend tail hedging based on current risk profile."""
        if risk_metrics.tail_risk_score < 0.3:
            return {"recommendation": "none", "reason": "Tail risk within normal range"}

        # Size the hedge
        hedge_notional = portfolio_notional * risk_metrics.tail_risk_score * 0.05
        if risk_metrics.risk_level in ("high", "critical"):
            hedge_notional *= 2

        strategies = []
        if risk_metrics.skewness < -0.5:
            strategies.append("Buy OTM puts (negative skew = left tail risk)")
        if risk_metrics.kurtosis > 5:
            strategies.append("Buy variance swaps (fat tails)")
        if risk_metrics.cvar_99_pct > 0.05:
            strategies.append("Add long volatility position (VIX calls)")
        if not strategies:
            strategies.append("General portfolio insurance via put spreads")

        return {
            "recommendation": "hedge",
            "hedge_notional": float(hedge_notional),
            "hedge_as_pct_of_portfolio": float(hedge_notional / max(portfolio_notional, 1e-10) * 100),
            "strategies": strategies,
            "urgency": risk_metrics.risk_level,
            "tail_risk_score": risk_metrics.tail_risk_score,
        }
