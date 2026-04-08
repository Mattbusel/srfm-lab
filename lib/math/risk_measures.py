"""
Advanced risk measures for portfolio and trading.

Implements:
  - VaR and CVaR (historical, parametric, Monte Carlo)
  - Expected Shortfall with confidence intervals
  - Conditional Drawdown at Risk (CDaR)
  - Maximum Drawdown statistics
  - Omega ratio, Calmar ratio, Sortino, Ulcer Index
  - Tail risk metrics: LPM, HPM, Kappa ratio
  - Risk contribution and marginal risk
  - Coherent risk measures (CVaR is coherent, VaR is not)
  - Stress testing: scenario PnL distribution
  - Risk attribution by factor
  - Rolling risk metrics time series
  - Pain ratio and pain index
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── VaR and CVaR ─────────────────────────────────────────────────────────────

def historical_var(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Historical VaR at confidence level (1-alpha)."""
    return float(-np.percentile(returns, 100 * alpha))


def historical_cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Historical CVaR (Expected Shortfall) at confidence level (1-alpha)."""
    threshold = np.percentile(returns, 100 * alpha)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return float(-returns.min())
    return float(-tail.mean())


def parametric_var(
    returns: np.ndarray,
    alpha: float = 0.05,
    distribution: str = "normal",
) -> float:
    """
    Parametric VaR under Normal or Student-t distribution.
    """
    mu = float(returns.mean())
    sigma = float(returns.std())

    if distribution == "normal":
        from scipy.stats import norm
        z = norm.ppf(alpha)
        return float(-(mu + z * sigma))
    elif distribution == "t":
        from scipy.stats import t
        # Fit Student-t
        df, loc, scale = t.fit(returns)
        z = t.ppf(alpha, df=df, loc=loc, scale=scale)
        return float(-z)
    else:
        return historical_var(returns, alpha)


def monte_carlo_var(
    returns: np.ndarray,
    alpha: float = 0.05,
    n_sim: int = 10000,
    method: str = "bootstrap",
    seed: int = 42,
) -> dict:
    """
    Monte Carlo VaR estimation.
    method: 'bootstrap' (resample) or 'parametric' (simulate from fitted distribution)
    """
    rng = np.random.default_rng(seed)
    n = len(returns)

    if method == "bootstrap":
        sim_returns = rng.choice(returns, size=(n_sim, n), replace=True).sum(axis=1)
    else:
        mu, sigma = returns.mean(), returns.std()
        sim_returns = rng.normal(mu * n, sigma * math.sqrt(n), n_sim)

    var = float(-np.percentile(sim_returns, 100 * alpha))
    cvar = float(-sim_returns[sim_returns <= -var].mean()) if (sim_returns <= -var).any() else var

    # Confidence interval via bootstrap
    n_boot = 200
    var_boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_sim, n_sim)
        var_boots.append(-np.percentile(sim_returns[idx], 100 * alpha))
    var_ci_lo = float(np.percentile(var_boots, 5))
    var_ci_hi = float(np.percentile(var_boots, 95))

    return {
        "var": var,
        "cvar": cvar,
        "var_ci_90": (var_ci_lo, var_ci_hi),
    }


# ── Drawdown Analysis ─────────────────────────────────────────────────────────

def drawdown_series(returns: np.ndarray) -> np.ndarray:
    """Compute drawdown time series from returns."""
    wealth = np.exp(np.cumsum(np.log(1 + returns)))
    rolling_max = np.maximum.accumulate(wealth)
    dd = (wealth - rolling_max) / rolling_max
    return dd


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown from return series."""
    dd = drawdown_series(returns)
    return float(dd.min())


def drawdown_statistics(returns: np.ndarray) -> dict:
    """Full drawdown statistics."""
    dd = drawdown_series(returns)
    mdd = float(dd.min())

    # Find drawdown periods
    in_dd = dd < 0
    dd_periods = []
    current_start = None
    for t in range(len(dd)):
        if in_dd[t] and current_start is None:
            current_start = t
        elif not in_dd[t] and current_start is not None:
            dd_periods.append((current_start, t, float(dd[current_start:t].min())))
            current_start = None
    if current_start is not None:
        dd_periods.append((current_start, len(dd), float(dd[current_start:].min())))

    durations = [p[1] - p[0] for p in dd_periods]
    depths = [p[2] for p in dd_periods]

    return {
        "max_drawdown": mdd,
        "avg_drawdown": float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0,
        "n_drawdowns": len(dd_periods),
        "avg_duration": float(np.mean(durations)) if durations else 0.0,
        "max_duration": int(max(durations)) if durations else 0,
        "avg_depth": float(np.mean(depths)) if depths else 0.0,
        "time_underwater_pct": float((dd < 0).mean()),
    }


def conditional_drawdown_at_risk(
    returns: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """
    CDaR: Expected Shortfall of the drawdown distribution.
    Average of the alpha worst drawdowns.
    """
    dd = drawdown_series(returns)
    threshold = np.percentile(dd, 100 * alpha)
    worst_dds = dd[dd <= threshold]
    if len(worst_dds) == 0:
        return float(abs(dd.min()))
    return float(abs(worst_dds.mean()))


# ── Return-Based Risk Metrics ─────────────────────────────────────────────────

def sortino_ratio(
    returns: np.ndarray,
    target: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Sortino ratio: excess return over downside deviation."""
    excess = returns - target / periods_per_year
    downside = np.sqrt(np.mean(np.minimum(excess, 0)**2) + 1e-10)
    return float(excess.mean() / downside * math.sqrt(periods_per_year))


def calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Calmar ratio: annualized return / max drawdown."""
    ann_return = float(returns.mean() * periods_per_year)
    mdd = abs(max_drawdown(returns))
    return float(ann_return / max(mdd, 1e-10))


def omega_ratio(
    returns: np.ndarray,
    target: float = 0.0,
) -> float:
    """Omega ratio: gains / losses relative to threshold."""
    gains = float(np.sum(np.maximum(returns - target, 0)))
    losses = float(np.sum(np.maximum(target - returns, 0)))
    return float(gains / max(losses, 1e-10))


def ulcer_index(returns: np.ndarray) -> float:
    """
    Ulcer Index: RMS of drawdowns.
    Penalizes deep, prolonged drawdowns.
    """
    dd = drawdown_series(returns)
    return float(math.sqrt(np.mean(dd**2)))


def pain_index(returns: np.ndarray) -> float:
    """Pain Index: average absolute drawdown."""
    dd = drawdown_series(returns)
    return float(abs(dd).mean())


def pain_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Pain ratio: annualized return / pain index."""
    ann_return = float(returns.mean() * periods_per_year)
    pi = pain_index(returns)
    return float(ann_return / max(pi, 1e-10))


def lower_partial_moment(
    returns: np.ndarray,
    target: float = 0.0,
    n: int = 2,
) -> float:
    """LPM_n: E[max(target - r, 0)^n]."""
    shortfalls = np.maximum(target - returns, 0)
    return float(np.mean(shortfalls**n))


def higher_partial_moment(
    returns: np.ndarray,
    target: float = 0.0,
    n: int = 2,
) -> float:
    """HPM_n: E[max(r - target, 0)^n]."""
    surpluses = np.maximum(returns - target, 0)
    return float(np.mean(surpluses**n))


def kappa_ratio(
    returns: np.ndarray,
    target: float = 0.0,
    n: int = 3,
    periods_per_year: int = 252,
) -> float:
    """Kappa n: (annualized return - target) / LPM_n^{1/n}."""
    excess = float(returns.mean() * periods_per_year) - target
    lpm = lower_partial_moment(returns, target / periods_per_year, n)
    return float(excess / max(lpm**(1.0/n), 1e-10))


# ── Risk Contribution ─────────────────────────────────────────────────────────

def marginal_var(
    weights: np.ndarray,
    returns_matrix: np.ndarray,
    alpha: float = 0.05,
    delta: float = 0.001,
) -> np.ndarray:
    """
    Marginal VaR: change in portfolio VaR per unit increase in position.
    Computed via finite differences.
    """
    n = len(weights)
    port_returns = returns_matrix @ weights
    base_var = historical_var(port_returns, alpha)
    mvar = np.zeros(n)

    for i in range(n):
        w_bump = weights.copy()
        w_bump[i] += delta
        w_bump /= w_bump.sum()
        port_bump = returns_matrix @ w_bump
        mvar[i] = (historical_var(port_bump, alpha) - base_var) / delta

    return mvar


def component_var(
    weights: np.ndarray,
    returns_matrix: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Component VaR: decompose portfolio VaR into contributions per asset.
    """
    mvar = marginal_var(weights, returns_matrix, alpha)
    cvar = weights * mvar

    total_var = historical_var(returns_matrix @ weights, alpha)
    pct_contribution = cvar / max(total_var, 1e-10)

    return {
        "marginal_var": mvar,
        "component_var": cvar,
        "portfolio_var": float(total_var),
        "pct_contribution": pct_contribution,
        "diversification_ratio": float(
            (np.abs(weights) @ np.sqrt(np.diag(np.cov(returns_matrix.T))))
            / max(returns_matrix @ weights).std()
        ) if len(weights) > 1 else 1.0,
    }


# ── Stress Testing ────────────────────────────────────────────────────────────

@dataclass
class StressScenario:
    name: str
    return_shocks: np.ndarray    # per-asset return in this scenario
    probability: float           # scenario weight


def scenario_var(
    weights: np.ndarray,
    scenarios: list[StressScenario],
) -> dict:
    """
    VaR and CVaR from a set of stress scenarios.
    """
    pnls = np.array([
        float(weights @ s.return_shocks) for s in scenarios
    ])
    probs = np.array([s.probability for s in scenarios])
    probs /= probs.sum()

    # Sort by PnL
    idx = np.argsort(pnls)
    sorted_pnl = pnls[idx]
    sorted_probs = probs[idx]
    cum_probs = np.cumsum(sorted_probs)

    # Find 5% VaR
    alpha = 0.05
    var_idx = np.searchsorted(cum_probs, alpha)
    var_idx = min(var_idx, len(sorted_pnl) - 1)
    var = float(-sorted_pnl[var_idx])

    # CVaR
    tail_mask = cum_probs <= alpha
    if tail_mask.any():
        tail_pnl = sorted_pnl[tail_mask]
        tail_probs = sorted_probs[tail_mask]
        cvar = float(-(tail_pnl * tail_probs).sum() / tail_probs.sum())
    else:
        cvar = var

    worst_scenario = scenarios[idx[0]].name

    return {
        "scenario_var_5pct": var,
        "scenario_cvar_5pct": cvar,
        "worst_scenario": worst_scenario,
        "worst_pnl": float(sorted_pnl[0]),
        "scenario_pnls": {s.name: float(weights @ s.return_shocks) for s in scenarios},
    }


# ── Rolling Risk Metrics ──────────────────────────────────────────────────────

def rolling_risk_metrics(
    returns: np.ndarray,
    window: int = 63,
    periods_per_year: int = 252,
) -> dict:
    """
    Rolling window risk metrics as time series.
    """
    T = len(returns)
    result = {
        "var_5pct": np.zeros(T),
        "cvar_5pct": np.zeros(T),
        "volatility": np.zeros(T),
        "max_drawdown": np.zeros(T),
        "sortino": np.zeros(T),
        "sharpe": np.zeros(T),
    }

    for t in range(window, T):
        r_win = returns[t - window: t]
        result["var_5pct"][t] = historical_var(r_win)
        result["cvar_5pct"][t] = historical_cvar(r_win)
        result["volatility"][t] = float(r_win.std() * math.sqrt(periods_per_year))
        result["max_drawdown"][t] = abs(max_drawdown(r_win))
        result["sortino"][t] = sortino_ratio(r_win, periods_per_year=periods_per_year)
        mu = float(r_win.mean())
        sigma = float(r_win.std())
        result["sharpe"][t] = float(mu / max(sigma, 1e-10) * math.sqrt(periods_per_year))

    return result


# ── Full Risk Report ──────────────────────────────────────────────────────────

def risk_report(returns: np.ndarray, periods_per_year: int = 252) -> dict:
    """Generate comprehensive risk report for a return series."""
    dd_stats = drawdown_statistics(returns)
    mu = float(returns.mean() * periods_per_year)
    sigma = float(returns.std() * math.sqrt(periods_per_year))

    return {
        "annualized_return": mu,
        "annualized_vol": sigma,
        "sharpe_ratio": float(mu / max(sigma, 1e-10)),
        "sortino_ratio": sortino_ratio(returns, periods_per_year=periods_per_year),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "omega_ratio": omega_ratio(returns),
        "ulcer_index": ulcer_index(returns),
        "pain_ratio": pain_ratio(returns, periods_per_year),
        "kappa_3": kappa_ratio(returns, n=3, periods_per_year=periods_per_year),
        "var_1pct": historical_var(returns, 0.01),
        "var_5pct": historical_var(returns, 0.05),
        "cvar_1pct": historical_cvar(returns, 0.01),
        "cvar_5pct": historical_cvar(returns, 0.05),
        "cdar_5pct": conditional_drawdown_at_risk(returns, 0.05),
        "skewness": float(
            np.mean(((returns - returns.mean()) / (returns.std() + 1e-10))**3)
        ),
        "excess_kurtosis": float(
            np.mean(((returns - returns.mean()) / (returns.std() + 1e-10))**4) - 3
        ),
        **dd_stats,
    }
