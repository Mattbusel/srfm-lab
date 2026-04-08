"""
Market making and optimal execution mathematics.

Implements:
  - Avellaneda-Stoikov optimal spread (inventory-aware market making)
  - Almgren-Chriss optimal execution schedule
  - Glosten-Milgrom model (adverse selection in spreads)
  - Inventory risk management
  - TWAP/VWAP schedule optimization
  - Linear impact model with decay
  - Square-root market impact
  - Implementation shortfall minimization
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Avellaneda-Stoikov ────────────────────────────────────────────────────────

@dataclass
class AvellanedaStoikovParams:
    sigma: float = 0.02        # volatility per unit time
    gamma: float = 0.1         # risk aversion
    k: float = 1.5             # order arrival intensity parameter
    A: float = 140.0           # order arrival rate constant
    T: float = 1.0             # total trading horizon
    dt: float = 1.0 / 86400   # time step (1 second / 1 day)


def avellaneda_stoikov_quotes(
    params: AvellanedaStoikovParams,
    q: float,         # current inventory
    t: float,         # current time (0 to T)
    S: float,         # mid price
) -> tuple[float, float, float]:
    """
    Compute optimal bid/ask quotes for a market maker.

    Returns (r*, bid, ask) where:
      r* = reservation price (indifference price accounting for inventory)
      bid = optimal bid quote
      ask = optimal ask quote
    """
    p = params
    tau = p.T - t  # time remaining

    # Reservation price: shift mid by inventory risk
    r_star = S - q * p.gamma * p.sigma ** 2 * tau

    # Optimal spread
    spread = p.gamma * p.sigma ** 2 * tau + (2 / p.gamma) * math.log(1 + p.gamma / p.k)
    half_spread = spread / 2

    bid = r_star - half_spread
    ask = r_star + half_spread

    return r_star, bid, ask


def as_inventory_pnl(
    params: AvellanedaStoikovParams,
    inventory_path: np.ndarray,
    time_path: np.ndarray,
) -> float:
    """Expected PnL penalty from inventory under AS model."""
    total_penalty = 0.0
    for q, t in zip(inventory_path, time_path):
        tau = params.T - t
        total_penalty += params.gamma * params.sigma ** 2 * tau * q ** 2
    return -float(total_penalty / len(inventory_path))


# ── Almgren-Chriss ────────────────────────────────────────────────────────────

@dataclass
class AlmgrenChrissParams:
    sigma: float = 0.02     # daily vol
    eta: float = 2.5e-7     # temporary impact coefficient
    gamma: float = 2.5e-8   # permanent impact coefficient
    epsilon: float = 0.0    # fixed cost per trade
    tau: float = 1 / 252    # time per step (1 trading day)
    risk_aversion: float = 1e-6


def almgren_chriss_schedule(
    params: AlmgrenChrissParams,
    X: float,           # total shares to execute
    T: int,             # number of periods
) -> np.ndarray:
    """
    Optimal execution schedule (Almgren-Chriss 2001).
    Returns array of shares to sell in each period.

    Balances market impact (execute fast) vs timing risk (execute slowly).
    """
    p = params
    kappa_bar = math.sqrt(p.risk_aversion * p.sigma ** 2 / p.eta)
    kappa = math.acosh(0.5 * (kappa_bar ** 2 * p.tau ** 2 + 2) + 1e-10) / p.tau

    # Optimal trajectory: x(t) = X * sinh(kappa*(T-t)) / sinh(kappa*T)
    t_grid = np.arange(T + 1)
    denom = math.sinh(kappa * T * p.tau)
    if denom < 1e-10:
        return np.full(T, X / T)

    trajectory = X * np.sinh(kappa * (T - t_grid) * p.tau) / denom
    schedule = -np.diff(trajectory)  # shares sold each period
    return np.maximum(schedule, 0)


def almgren_chriss_expected_cost(
    params: AlmgrenChrissParams,
    schedule: np.ndarray,
    X: float,
) -> dict:
    """
    Expected cost and variance of an execution schedule.
    """
    p = params
    T = len(schedule)
    n_j = schedule  # shares in each period

    # Permanent impact cost
    perm_cost = p.gamma * X ** 2 / 2

    # Temporary impact cost
    temp_cost = sum(p.eta * (n_j[t] / p.tau) ** 2 * p.tau for t in range(T))

    # Timing risk: variance of execution
    timing_var = sum(p.sigma ** 2 * p.tau * sum(n_j[t:]) ** 2 for t in range(T))

    return {
        "permanent_impact_cost": float(perm_cost),
        "temporary_impact_cost": float(temp_cost),
        "total_expected_cost": float(perm_cost + temp_cost),
        "timing_variance": float(timing_var),
        "expected_shortfall": float(perm_cost + temp_cost),
    }


# ── Square-root market impact ──────────────────────────────────────────────────

def sqrt_impact_model(
    order_size: float,
    daily_volume: float,
    sigma_daily: float,
    bid_ask_spread: float = 0.0,
    alpha: float = 0.5,    # participation rate exponent (0.5 = square root)
) -> float:
    """
    Square-root market impact model (widespread empirical finding).
    Impact = sigma * sqrt(order_size / daily_volume)

    Returns expected price impact as fraction of price.
    """
    participation_rate = order_size / max(daily_volume, 1.0)
    impact = sigma_daily * (participation_rate ** alpha)
    return float(impact + bid_ask_spread / 2)


def optimal_participation_rate(
    order_size: float,
    daily_volume: float,
    sigma_daily: float,
    urgency: float = 0.5,    # 0=minimize impact, 1=execute immediately
) -> float:
    """
    Optimal participation rate given urgency.
    urgency=0: spread trade over full day (min impact)
    urgency=1: execute immediately (max impact)
    """
    # Min impact: sqrt(Q/V) is minimized by slower execution
    # Tradeoff: slower = more timing risk
    min_rate = min(sqrt_impact_model(order_size, daily_volume, sigma_daily), 0.20)
    return float(urgency + (1 - urgency) * min_rate)


# ── Glosten-Milgrom spread decomposition ──────────────────────────────────────

def glosten_milgrom_spread(
    mu: float,      # prob of informed trader
    delta: float,   # E[|V - S| / S] for informed traders (info advantage)
    V: float,       # fundamental value
    S: float,       # current mid price
) -> tuple[float, float]:
    """
    Glosten-Milgrom equilibrium spread.
    Uninformed market maker sets bid/ask to break even.

    Returns (bid, ask).
    E[profit from uninformed] = E[loss to informed]
    Spread = 2 * mu * delta / (1 - mu)
    """
    half_spread = mu * delta * S / max(1 - mu, 0.01)
    bid = S - half_spread
    ask = S + half_spread
    return bid, ask


def adverse_selection_component(
    prices: np.ndarray,
    trades_dir: np.ndarray,    # +1 = buy, -1 = sell
    window: int = 50,
) -> float:
    """
    Estimate adverse selection component of the spread.
    = fraction of spread due to information asymmetry.
    Uses price revision after trades.
    """
    n = min(len(prices) - 1, len(trades_dir))
    if n < window:
        return 0.0

    # Price change 5 bars after trade
    price_changes = np.array([
        prices[i + 5] - prices[i]
        for i in range(n - 5)
    ])
    directions = trades_dir[:n - 5]

    # Adverse selection = covariance(price_change, trade_direction)
    cov = float(np.cov(price_changes, directions)[0, 1])
    spread_proxy = float(np.abs(np.diff(prices)).mean())
    return float(min(abs(cov) / max(spread_proxy, 1e-10), 1.0))


# ── TWAP/VWAP schedules ───────────────────────────────────────────────────────

def twap_schedule(total_shares: float, n_periods: int) -> np.ndarray:
    """Equal time-weighted schedule."""
    return np.full(n_periods, total_shares / n_periods)


def vwap_schedule(
    total_shares: float,
    historical_volume_profile: np.ndarray,
) -> np.ndarray:
    """
    Volume-weighted schedule: trade in proportion to expected intraday volume.
    """
    vol_frac = historical_volume_profile / historical_volume_profile.sum()
    return total_shares * vol_frac


def implementation_shortfall(
    arrival_price: float,
    execution_prices: np.ndarray,
    shares: np.ndarray,
) -> dict:
    """
    Implementation shortfall = paper portfolio return - actual portfolio return.
    Measures total execution quality vs decision price.
    """
    total_shares = shares.sum()
    if total_shares == 0:
        return {"total_is_bps": 0.0, "market_impact_bps": 0.0}

    avg_execution = float(np.average(execution_prices, weights=shares))
    is_pct = (avg_execution - arrival_price) / arrival_price  # positive = paid more
    is_bps = is_pct * 10000

    return {
        "arrival_price": float(arrival_price),
        "avg_execution_price": float(avg_execution),
        "total_is_bps": float(is_bps),
        "total_shares": float(total_shares),
        "slippage_per_share": float(avg_execution - arrival_price),
    }
