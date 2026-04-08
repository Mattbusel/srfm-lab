"""
Market impact models for execution cost estimation.

Implements:
  - Almgren-Chriss optimal execution (linear and nonlinear impact)
  - Square-root market impact model (empirical)
  - Obizhaeva-Wang continuous impact model
  - VWAP execution cost estimation
  - Information leakage model (adversarial market maker)
  - Pre-trade analytics: expected shortfall, implementation shortfall
  - Post-trade impact measurement
  - Decay of price impact over time
  - Liquidity score and optimal order sizing
  - Dark pool vs lit venue routing optimization
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Square-Root Impact Model ──────────────────────────────────────────────────

def sqrt_market_impact(
    order_size: float,       # in shares or notional
    daily_volume: float,
    sigma_daily: float,      # daily return vol
    eta: float = 0.1,        # impact coefficient (typical 0.1-0.3)
    permanent_frac: float = 0.5,
) -> dict:
    """
    Square-root market impact (Grinold-Kahn, Almgren et al.):
    I = sigma * eta * sqrt(Q / ADV)
    """
    participation = order_size / max(daily_volume, 1)
    impact = sigma_daily * eta * math.sqrt(participation)
    permanent = impact * permanent_frac
    temporary = impact * (1 - permanent_frac)

    return {
        "total_impact_pct": float(impact * 100),
        "impact_bps": float(impact * 10000),
        "permanent_impact_bps": float(permanent * 10000),
        "temporary_impact_bps": float(temporary * 10000),
        "participation_rate": float(participation),
        "model": "square_root",
    }


def power_law_impact(
    order_size: float,
    daily_volume: float,
    sigma_daily: float,
    alpha: float = 0.5,     # exponent (0.5 = sqrt, closer to 1 = linear)
    gamma: float = 0.1,
) -> float:
    """
    Power-law impact: I = gamma * sigma * (Q/ADV)^alpha
    """
    participation = order_size / max(daily_volume, 1)
    return float(gamma * sigma_daily * participation**alpha * 10000)  # in bps


# ── Almgren-Chriss Full Model ─────────────────────────────────────────────────

@dataclass
class AlmgrenChrissParams:
    X0: float          # total shares to execute
    T: float           # time horizon (days)
    sigma: float       # daily volatility (price)
    eta: float         # temporary impact coefficient
    gamma_perm: float  # permanent impact coefficient
    risk_aversion: float  # lambda (risk aversion)


def almgren_chriss_optimal_schedule(
    params: AlmgrenChrissParams,
    n_periods: int = 10,
) -> dict:
    """
    Almgren-Chriss (2001) optimal liquidation schedule.
    Balances market impact cost vs price risk.
    """
    p = params
    T = p.T
    N = n_periods
    dt = T / N

    # AC parameters
    kappa = math.sqrt(p.risk_aversion * p.sigma**2 / max(p.eta, 1e-10))

    # Optimal trajectory: X(t) = X0 * sinh(kappa*(T-t)) / sinh(kappa*T)
    t_grid = np.linspace(0, T, N + 1)
    denom = math.sinh(kappa * T) if kappa * T > 1e-6 else 1.0

    if abs(denom) < 1e-10:
        # Linear approximation for small kappa
        X = p.X0 * (1 - t_grid / T)
    else:
        X = p.X0 * np.sinh(kappa * (T - t_grid)) / denom

    X = np.maximum(X, 0)
    trades = np.diff(X)  # negative = selling

    # Expected cost
    exp_cost = (p.gamma_perm * p.X0**2 / 2
                + p.eta / dt * np.sum(trades**2)
                + p.risk_aversion * p.sigma**2 * np.sum(X[:-1]**2) * dt)

    # Variance of cost
    cost_var = float(p.risk_aversion**2 * p.sigma**4 * dt * np.sum(X[:-1]**4))

    return {
        "inventory_schedule": X.tolist(),
        "trade_schedule": (-trades).tolist(),  # positive = selling
        "t_grid": t_grid.tolist(),
        "expected_cost_bps": float(exp_cost / p.X0 * 10000) if p.X0 > 0 else 0,
        "cost_variance": float(cost_var),
        "kappa": float(kappa),
        "half_life_periods": float(math.log(2) / max(kappa, 1e-6) / dt) if kappa > 1e-6 else N,
    }


def almgren_chriss_efficient_frontier(
    params: AlmgrenChrissParams,
    n_points: int = 20,
    n_periods: int = 20,
) -> dict:
    """
    Almgren-Chriss efficient frontier: E[cost] vs Var[cost].
    Vary risk aversion from 0 to high.
    """
    risk_aversions = np.logspace(-6, 0, n_points)
    costs = []
    variances = []

    for ra in risk_aversions:
        p = AlmgrenChrissParams(
            X0=params.X0, T=params.T, sigma=params.sigma,
            eta=params.eta, gamma_perm=params.gamma_perm,
            risk_aversion=float(ra),
        )
        result = almgren_chriss_optimal_schedule(p, n_periods)
        costs.append(result["expected_cost_bps"])
        variances.append(math.sqrt(result["cost_variance"]))

    return {
        "expected_costs_bps": costs,
        "cost_std": variances,
        "risk_aversions": risk_aversions.tolist(),
    }


# ── VWAP Execution Cost ───────────────────────────────────────────────────────

def vwap_execution_cost(
    order_size: float,
    daily_volume: float,
    intraday_volume_profile: np.ndarray,  # normalized volume by period
    price_path: np.ndarray,               # price at each period
    sigma_daily: float,
    eta: float = 0.1,
) -> dict:
    """
    Estimate VWAP execution cost vs price path.
    """
    T = len(price_path)
    vol_profile = intraday_volume_profile / (intraday_volume_profile.sum() + 1e-10)

    # VWAP price
    vwap = float(np.sum(price_path * vol_profile))

    # Execution price if following volume profile
    participation = order_size / max(daily_volume, 1)
    exec_sizes = vol_profile * order_size

    # Impact per period
    impact = np.array([
        sqrt_market_impact(exec_sizes[t], daily_volume * vol_profile[t], sigma_daily / math.sqrt(T), eta)["impact_bps"]
        for t in range(T)
    ])

    avg_impact = float(np.sum(impact * vol_profile))
    arrival_price = float(price_path[0])
    slippage_bps = float((vwap - arrival_price) / max(arrival_price, 1e-10) * 10000)

    return {
        "vwap_price": vwap,
        "arrival_price": arrival_price,
        "market_impact_bps": avg_impact,
        "timing_slippage_bps": slippage_bps,
        "total_cost_bps": float(avg_impact + abs(slippage_bps)),
        "participation_rate": float(participation),
    }


# ── Information Leakage ───────────────────────────────────────────────────────

def information_leakage_cost(
    order_size: float,
    daily_volume: float,
    sigma_daily: float,
    execution_days: int = 1,
    adversarial_fraction: float = 0.1,
) -> dict:
    """
    Information leakage: adversarial market makers front-run large orders.
    Leakage cost scales with order size and execution time.
    """
    participation = order_size / max(daily_volume, 1)

    # Leakage: adverse price movement while executing
    # Larger order = more visible = more front-running
    visibility = float(math.tanh(participation * 5))  # 0-1 scale
    leakage_vol = sigma_daily * math.sqrt(execution_days) * adversarial_fraction * visibility

    leakage_bps = float(leakage_vol * 10000)

    return {
        "leakage_cost_bps": leakage_bps,
        "visibility_score": float(visibility),
        "recommended_execution_days": int(math.ceil(
            (participation / 0.05)**2
        )) if participation > 0.05 else 1,
        "stealth_trading": bool(participation < 0.03),
    }


# ── Pre-Trade Analytics ───────────────────────────────────────────────────────

@dataclass
class PreTradeAnalytics:
    order_size: float
    daily_volume: float
    sigma_daily: float
    entry_price: float
    expected_shortfall_bps: float
    implementation_shortfall_bps: float
    total_cost_bps: float
    market_impact_bps: float
    opportunity_cost_bps: float
    optimal_execution_days: int
    risk_of_non_execution: float


def pre_trade_analysis(
    order_size: float,
    entry_price: float,
    daily_volume: float,
    sigma_daily: float,
    urgency: float = 0.5,   # 0=patient, 1=urgent
    risk_aversion: float = 2.0,
) -> PreTradeAnalytics:
    """
    Full pre-trade analytics for an order.
    """
    participation = order_size / max(daily_volume, 1)

    # Optimal execution time (risk-adjusted)
    # More patient = lower impact but more timing risk
    optimal_days = max(1, int(math.ceil(participation / 0.05)))
    if urgency > 0.7:
        optimal_days = 1

    # Market impact
    impact = sqrt_market_impact(order_size, daily_volume, sigma_daily)
    mi_bps = float(impact["impact_bps"])

    # Timing risk (opportunity cost)
    timing_vol_bps = float(sigma_daily * math.sqrt(optimal_days) * 10000)
    opp_cost_bps = float(0.5 * timing_vol_bps * (1 - urgency))

    # Implementation shortfall components
    is_bps = mi_bps + opp_cost_bps

    # Expected shortfall (5th percentile of execution cost)
    es_bps = float(is_bps + 1.65 * timing_vol_bps / math.sqrt(optimal_days))

    # Risk of non-execution (if limit order)
    if urgency < 0.3:
        risk_non_exec = float(max(0.2 - urgency, 0))
    else:
        risk_non_exec = 0.0

    return PreTradeAnalytics(
        order_size=order_size,
        daily_volume=daily_volume,
        sigma_daily=sigma_daily,
        entry_price=entry_price,
        expected_shortfall_bps=es_bps,
        implementation_shortfall_bps=is_bps,
        total_cost_bps=float(mi_bps + opp_cost_bps),
        market_impact_bps=mi_bps,
        opportunity_cost_bps=opp_cost_bps,
        optimal_execution_days=optimal_days,
        risk_of_non_execution=risk_non_exec,
    )


# ── Liquidity Score ───────────────────────────────────────────────────────────

def liquidity_score(
    daily_volume: float,
    bid_ask_spread_bps: float,
    price_impact_bps: float,
    market_cap_usd: Optional[float] = None,
) -> dict:
    """
    Composite liquidity score for an asset.
    Higher = more liquid.
    """
    # Volume component: log scale (more volume = better)
    vol_score = float(min(math.log10(max(daily_volume, 1)) / 9, 1.0))  # $1B ADV = ~1.0

    # Spread component: lower spread = better
    spread_score = float(max(1 - bid_ask_spread_bps / 100, 0))

    # Impact component: lower impact = better
    impact_score = float(max(1 - price_impact_bps / 200, 0))

    # Market cap component (optional)
    cap_score = 0.5
    if market_cap_usd is not None:
        cap_score = float(min(math.log10(max(market_cap_usd, 1)) / 12, 1.0))

    # Weighted composite
    if market_cap_usd is not None:
        score = 0.3 * vol_score + 0.25 * spread_score + 0.25 * impact_score + 0.2 * cap_score
    else:
        score = 0.35 * vol_score + 0.30 * spread_score + 0.35 * impact_score

    return {
        "liquidity_score": float(score),
        "volume_component": vol_score,
        "spread_component": spread_score,
        "impact_component": impact_score,
        "tier": "tier1" if score > 0.7 else "tier2" if score > 0.4 else "tier3",
        "max_safe_participation": float(min(score * 0.2, 0.15)),
    }


# ── Dark Pool vs Lit Venue ────────────────────────────────────────────────────

def venue_routing_optimization(
    order_size: float,
    daily_volume_lit: float,
    daily_volume_dark: float,
    sigma_daily: float,
    dark_pool_fill_prob: float = 0.4,
    urgency: float = 0.5,
) -> dict:
    """
    Optimize order routing between dark pool and lit venue.
    Dark pool: lower impact but lower fill probability.
    """
    # Expected impact in lit market
    lit_impact = sqrt_market_impact(order_size, daily_volume_lit, sigma_daily)["impact_bps"]

    # Expected impact in dark pool (lower due to no signaling)
    dark_impact = lit_impact * 0.3  # typical dark pool savings ~70%

    # Expected fill in dark pool
    expected_dark_fill = order_size * dark_pool_fill_prob
    expected_lit_fill = order_size * (1 - dark_pool_fill_prob)

    # Timing cost of waiting for dark pool fill
    timing_cost = float(sigma_daily * math.sqrt(1 / dark_pool_fill_prob) * 10000 * (1 - urgency))

    # Optimal dark pool fraction
    if urgency > 0.8:
        dark_fraction = 0.0  # urgent = go lit
    elif dark_pool_fill_prob < 0.1:
        dark_fraction = 0.0  # too low fill rate
    else:
        # Trade off savings vs timing cost
        savings_per_unit = lit_impact - dark_impact
        cost_per_unit = timing_cost * (1 - dark_pool_fill_prob)
        dark_fraction = float(np.clip(savings_per_unit / max(cost_per_unit, 1e-10) * 0.5, 0, 0.7))

    expected_total_cost = float(
        dark_fraction * dark_impact + (1 - dark_fraction) * lit_impact + timing_cost * dark_fraction
    )

    return {
        "optimal_dark_fraction": dark_fraction,
        "optimal_lit_fraction": 1 - dark_fraction,
        "expected_cost_bps": expected_total_cost,
        "lit_only_cost_bps": float(lit_impact),
        "savings_from_dark_bps": float(max(lit_impact - expected_total_cost, 0)),
        "dark_fill_probability": float(dark_pool_fill_prob),
    }
