"""
Crash scenario generator — simulates market crash conditions for stress testing.

Generates:
  - Flash crash scenarios (rapid intraday move)
  - Systemic deleveraging (slow cascade)
  - Exchange hack / black swan event
  - Liquidity drought
  - Contagion cascade (cross-asset)
  - Post-FOMC panic
  - Stablecoin depeg event
  - Miner capitulation + long squeeze
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CrashScenario:
    name: str
    description: str
    price_path: np.ndarray          # price relative to starting price
    volume_path: np.ndarray         # volume multiplier vs normal
    oi_path: np.ndarray             # open interest multiplier vs normal
    funding_path: Optional[np.ndarray] = None
    volatility_path: Optional[np.ndarray] = None
    tags: list[str] = field(default_factory=list)
    max_drawdown: float = 0.0
    recovery_time: int = 0


def flash_crash_scenario(
    magnitude: float = 0.20,
    duration_minutes: int = 30,
    n_steps: int = 60,
    recovery_frac: float = 0.70,
    rng: Optional[np.random.Generator] = None,
) -> CrashScenario:
    """
    Flash crash: rapid <60 min drop followed by partial recovery.
    Typical: -20% in 30 min, +14% recovery (70% retracement).
    """
    rng = rng or np.random.default_rng(42)
    crash_steps = int(n_steps * duration_minutes / 60)
    recovery_steps = n_steps - crash_steps

    # Crash phase: exponential move down with noise
    t_crash = np.linspace(0, 1, crash_steps)
    crash_path = 1.0 - magnitude * t_crash**0.7 + rng.normal(0, magnitude * 0.05, crash_steps)

    # Recovery phase: partial bounce
    t_rec = np.linspace(0, 1, recovery_steps)
    bottom = crash_path[-1]
    recovery_target = bottom + (1.0 - bottom) * recovery_frac
    rec_path = bottom + (recovery_target - bottom) * (1 - np.exp(-3 * t_rec)) + rng.normal(0, 0.005, recovery_steps)

    price_path = np.concatenate([crash_path, rec_path])

    # Volume: spike during crash, elevated recovery
    vol_crash = 5.0 * np.exp(-t_crash * 2) + 2.0
    vol_rec = 1.5 * np.ones(recovery_steps)
    volume_path = np.concatenate([vol_crash, vol_rec])

    # OI: drops sharply during liquidations
    oi_crash = 1.0 - 0.4 * t_crash
    oi_rec = oi_crash[-1] * (1 + 0.1 * t_rec)
    oi_path = np.concatenate([oi_crash, oi_rec])

    # Funding: turns very negative during crash
    fund_crash = -0.05 * (1 - np.exp(-5 * t_crash))
    fund_rec = fund_crash[-1] * np.exp(-3 * t_rec)
    funding_path = np.concatenate([fund_crash, fund_rec])

    max_dd = float(1 - price_path.min())
    return CrashScenario(
        name="flash_crash",
        description=f"Flash crash: -{magnitude:.0%} in {duration_minutes}min, {recovery_frac:.0%} recovery",
        price_path=price_path,
        volume_path=volume_path,
        oi_path=oi_path,
        funding_path=funding_path,
        tags=["flash_crash", "liquidation", "high_vol"],
        max_drawdown=max_dd,
        recovery_time=recovery_steps,
    )


def systemic_deleveraging_scenario(
    total_drawdown: float = 0.50,
    duration_days: int = 60,
    n_steps: int = 300,
    bear_trap_count: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> CrashScenario:
    """
    Systemic deleveraging: slow grinding bear market with dead cat bounces.
    Typical: BTC -50% over 60 days with 2 bear traps.
    """
    rng = rng or np.random.default_rng(42)
    t = np.linspace(0, 1, n_steps)

    # Base trend: log-linear decline
    trend = 1.0 - total_drawdown * t**(0.8)

    # Add bear traps (30-40% bounces)
    for _ in range(bear_trap_count):
        trap_center = rng.uniform(0.2, 0.7)
        trap_width = 0.08
        trap_height = rng.uniform(0.10, 0.20)
        trap = trap_height * np.exp(-((t - trap_center) / trap_width)**2)
        trend = trend * (1 + trap * 0.3)

    # Add noise
    noise = rng.normal(0, 0.01, n_steps)
    price_path = np.maximum(trend + noise, 0.1)

    # Volume: declining trend with spikes on big down days
    ret = np.diff(np.log(price_path), prepend=np.log(price_path[0]))
    volume_path = 1.0 + 3.0 * np.abs(ret) * 20 + rng.exponential(0.1, n_steps)

    # OI: slowly declining as leverage unwound
    oi_path = 1.0 - 0.6 * t + rng.normal(0, 0.02, n_steps)
    oi_path = np.maximum(oi_path, 0.2)

    # Funding: persistently negative
    funding_path = -0.02 * (1 - np.exp(-2 * t)) + rng.normal(0, 0.003, n_steps)

    return CrashScenario(
        name="systemic_deleveraging",
        description=f"Systemic deleveraging: -{total_drawdown:.0%} over {duration_days}d with {bear_trap_count} bear traps",
        price_path=price_path,
        volume_path=volume_path,
        oi_path=oi_path,
        funding_path=funding_path,
        tags=["bear_market", "deleveraging", "bear_trap"],
        max_drawdown=float(1 - price_path.min()),
        recovery_time=n_steps,
    )


def black_swan_scenario(
    drop_magnitude: float = 0.40,
    n_steps: int = 120,
    rng: Optional[np.random.Generator] = None,
) -> CrashScenario:
    """
    Black swan: sudden large drop with no warning (exchange hack, protocol exploit, etc).
    Volume spikes massively, OI collapses, funding extreme negative.
    """
    rng = rng or np.random.default_rng(42)

    # Price: instant drop at step 5, then consolidation
    price_path = np.ones(n_steps)
    event_step = 5
    price_path[:event_step] = 1.0
    price_path[event_step:] = (1 - drop_magnitude) * (1 + rng.normal(0, 0.005, n_steps - event_step))
    price_path[event_step:event_step + 10] *= (1 - 0.1 * np.linspace(0, 1, 10))  # continued selling

    # Volume: 20x spike at event
    volume_path = rng.exponential(0.5, n_steps) + 1.0
    volume_path[event_step: event_step + 20] *= 20 * np.exp(-np.linspace(0, 3, 20))

    # OI: -60% at event
    oi_path = np.ones(n_steps)
    oi_path[event_step:] = 0.4 + 0.1 * np.linspace(0, 1, n_steps - event_step)

    # Funding: very negative
    funding_path = np.zeros(n_steps)
    funding_path[event_step:] = -0.08 * np.exp(-np.linspace(0, 2, n_steps - event_step))

    return CrashScenario(
        name="black_swan",
        description=f"Black swan event: instant -{drop_magnitude:.0%} with no warning",
        price_path=price_path,
        volume_path=volume_path,
        oi_path=oi_path,
        funding_path=funding_path,
        tags=["black_swan", "gap_down", "catastrophic"],
        max_drawdown=drop_magnitude,
        recovery_time=n_steps,
    )


def liquidity_drought_scenario(
    spread_multiple: float = 10.0,
    duration_steps: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> CrashScenario:
    """
    Liquidity drought: spreads widen massively, volume dries up, price gaps freely.
    Typical during major news events, weekends, or macro uncertainty.
    """
    rng = rng or np.random.default_rng(42)
    n = duration_steps

    # Price: more volatile due to thin book
    returns = rng.normal(-0.001, 0.02 * spread_multiple**0.3, n)
    price_path = np.exp(np.cumsum(returns))

    # Volume: 90% decline
    volume_path = 0.1 * np.ones(n) + rng.exponential(0.05, n)

    # OI: mildly declining
    oi_path = 1.0 - 0.2 * np.linspace(0, 1, n) + rng.normal(0, 0.02, n)

    return CrashScenario(
        name="liquidity_drought",
        description=f"Liquidity drought: spread {spread_multiple:.0f}x normal, 90% volume decline",
        price_path=price_path,
        volume_path=volume_path,
        oi_path=oi_path,
        tags=["liquidity", "wide_spread", "low_volume"],
        max_drawdown=float(1 - price_path.min()),
        recovery_time=n,
    )


def stablecoin_depeg_scenario(
    peg: float = 1.0,
    max_deviation: float = 0.15,
    n_steps: int = 150,
    rng: Optional[np.random.Generator] = None,
) -> CrashScenario:
    """
    Stablecoin depeg: algorithmic or collateral-backed stablecoin loses peg.
    Causes panic selling across crypto market.
    """
    rng = rng or np.random.default_rng(42)

    # Stablecoin price (deviation from peg)
    t = np.linspace(0, 1, n_steps)
    stable_price = peg - max_deviation * t**(0.5) * (1 + 0.3 * np.sin(15 * t))
    stable_price += rng.normal(0, 0.005, n_steps)

    # Crypto price: crashes as stablecoin confidence lost
    crypto_price = np.ones(n_steps)
    depeg_start = int(n_steps * 0.1)
    crypto_price[depeg_start:] = 1.0 - 0.35 * t[depeg_start:]**0.6

    # Volume: huge spike
    volume_path = 1 + 8 * t + rng.exponential(0.5, n_steps)

    oi_path = 1.0 - 0.5 * t

    return CrashScenario(
        name="stablecoin_depeg",
        description=f"Stablecoin depeg: -{max_deviation:.0%} from peg, crypto sells off {0.35:.0%}",
        price_path=crypto_price,
        volume_path=volume_path,
        oi_path=oi_path,
        tags=["stablecoin", "depeg", "contagion", "luna_type"],
        max_drawdown=float(1 - crypto_price.min()),
        recovery_time=n_steps,
    )


# ── Scenario Library ──────────────────────────────────────────────────────────

def get_scenario_library(rng: Optional[np.random.Generator] = None) -> dict[str, CrashScenario]:
    """Return all standard crash scenarios."""
    rng = rng or np.random.default_rng(42)
    return {
        "flash_crash": flash_crash_scenario(rng=rng),
        "flash_crash_severe": flash_crash_scenario(magnitude=0.35, rng=rng),
        "systemic_deleveraging": systemic_deleveraging_scenario(rng=rng),
        "bear_market_deep": systemic_deleveraging_scenario(total_drawdown=0.80, duration_days=300, rng=rng),
        "black_swan_mild": black_swan_scenario(drop_magnitude=0.25, rng=rng),
        "black_swan_severe": black_swan_scenario(drop_magnitude=0.55, rng=rng),
        "liquidity_drought": liquidity_drought_scenario(rng=rng),
        "stablecoin_depeg": stablecoin_depeg_scenario(rng=rng),
    }
