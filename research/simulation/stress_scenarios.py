"""
research/simulation/stress_scenarios.py

Pre-built stress test scenarios for the LARSA strategy. Each scenario
defines a price path transformation, vol modifier, and correlation
modifier that are applied on top of a base simulated market.

Scenarios are designed to probe specific failure modes:
  - BH false positives under choppy conditions
  - Drawdown survival under fat-tail events
  - Filter effectiveness during persistent non-trending periods
  - Recovery characteristics after extreme events

Usage
-----
>>> tester = StressTester()
>>> result = tester.run_scenario("flash_crash", strategy_fn=my_larsa_fn)
>>> result.worst_drawdown
-0.187
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from research.simulation.market_simulator import (
    GeometricBrownianMotion,
    OrnsteinUhlenbeck,
    RegimeSwitchingMarket,
    SimConfig,
    MarketRegime,
    DT_15M,
    _BARS_PER_DAY,
    _intraday_volume_factor,
)
from research.simulation.bh_signal_injector import (
    BHMassSimulator,
    compute_bh_mass_series,
    BH_FORM_DEFAULT,
    DEFAULT_CF,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StressResult:
    """Results from a single stress scenario run.

    Attributes
    ----------
    scenario_name:
        Name of the stress scenario applied.
    worst_drawdown:
        Maximum peak-to-trough drawdown (negative float, e.g. -0.20).
    recovery_bars:
        Number of bars from trough to prior peak recovery. -1 if not recovered.
    strategy_pnl:
        Total strategy PnL as fraction of initial capital (if strategy_fn provided).
    max_position_held:
        Maximum absolute position size seen during the scenario.
    signals_triggered:
        Number of bars on which bh_active was True.
    sharpe_ratio:
        Annualised Sharpe ratio of strategy PnL (NaN if no strategy).
    bh_false_positives:
        Count of BH activations that were not followed by directional move > 1%.
    price_path:
        The full price path (close prices) used in this scenario.
    strategy_returns:
        Per-bar strategy returns (empty if no strategy_fn provided).
    """
    scenario_name: str
    worst_drawdown: float
    recovery_bars: int
    strategy_pnl: float
    max_position_held: float
    signals_triggered: int
    sharpe_ratio: float
    bh_false_positives: int
    price_path: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    strategy_returns: NDArray[np.float64] = field(default_factory=lambda: np.array([]))


@dataclass
class StressScenario:
    """Definition of a named stress scenario.

    Attributes
    ----------
    name:
        Unique scenario identifier.
    description:
        Human-readable description of what this scenario tests.
    price_path_modifier:
        Callable (prices: np.ndarray, rng) -> np.ndarray that transforms the
        base simulated price path.
    vol_modifier:
        Multiplicative factor applied to base annual vol.
    correlation_modifier:
        Additive adjustment to inter-asset correlations (for multi-asset tests).
    duration_bars:
        Number of 15-min bars for the scenario.
    initial_price:
        Starting price level.
    base_annual_vol:
        Base annualised vol before vol_modifier is applied.
    seed:
        Random seed for reproducibility.
    """
    name: str
    description: str
    price_path_modifier: Callable[[NDArray[np.float64], np.random.Generator], NDArray[np.float64]]
    vol_modifier: float = 1.0
    correlation_modifier: float = 0.0
    duration_bars: int = 500
    initial_price: float = 100.0
    base_annual_vol: float = 0.20
    seed: int = 42


# ---------------------------------------------------------------------------
# Price path modifier helpers
# ---------------------------------------------------------------------------

def _apply_trend_modifier(
    prices: NDArray[np.float64],
    total_return: float,
    rng: np.random.Generator,
    noise_sigma: float = 0.005,
) -> NDArray[np.float64]:
    """Apply a sustained trend (total_return over all bars) to prices."""
    n = len(prices)
    t = np.linspace(0, 1, n)
    trend = np.exp(total_return * t)
    noise_z = rng.standard_normal(n)
    noise = np.exp(np.cumsum(noise_sigma * noise_z) - noise_sigma**2 * np.arange(n) / 2)
    modified = prices * trend * noise / noise[0]
    return np.maximum(modified, prices[0] * 0.001)


def _apply_flash_crash(
    prices: NDArray[np.float64],
    crash_depth: float,
    crash_bar: int,
    crash_bars: int,
    recovery_bars: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Apply a flash crash: sharp down move followed by recovery."""
    modified = prices.copy()
    n = len(modified)

    # Crash phase
    crash_end = min(crash_bar + crash_bars, n)
    for i in range(crash_bar, crash_end):
        t = (i - crash_bar) / max(crash_bars, 1)
        modified[i] = prices[crash_bar] * math.exp(crash_depth * t)

    # Recovery phase
    trough_price = modified[crash_end - 1]
    pre_crash_price = prices[crash_bar]
    rec_end = min(crash_end + recovery_bars, n)
    for i in range(crash_end, rec_end):
        t = (i - crash_end) / max(recovery_bars, 1)
        modified[i] = trough_price + (pre_crash_price - trough_price) * t
        # Add some noise during recovery
        modified[i] *= math.exp(rng.normal(0, 0.005))

    # Post-recovery: follow original path pattern but scaled from recovery level
    if rec_end < n:
        scale = modified[rec_end - 1] / max(prices[rec_end - 1], 1e-9)
        modified[rec_end:] = prices[rec_end:] * scale

    return np.maximum(modified, prices[0] * 0.001)


def _apply_chop(
    prices: NDArray[np.float64],
    rng: np.random.Generator,
    reversion_strength: float = 40.0,
) -> NDArray[np.float64]:
    """Apply a mean-reverting choppy regime -- kills trend following."""
    n = len(prices)
    dt = DT_15M
    sigma = float(np.std(np.diff(np.log(np.maximum(prices, 1e-12))))) / math.sqrt(dt)
    chop = OrnsteinUhlenbeck.generate(
        n - 1,
        kappa=reversion_strength,
        theta=prices[0],
        sigma=sigma * 0.5,
        x0=prices[0],
        dt=dt,
        seed=int(rng.integers(0, 2**31)),
    )
    return np.maximum(chop, prices[0] * 0.01)


def _apply_discontinuous_drop(
    prices: NDArray[np.float64],
    gaps: list[tuple[int, float]],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Apply discrete gap-down price events (liquidation cascades)."""
    modified = prices.copy()
    n = len(modified)
    cumulative_scale = 1.0
    for gap_bar, gap_pct in gaps:
        if gap_bar >= n:
            continue
        cumulative_scale *= (1.0 + gap_pct)
        # Apply from this bar forward
        modified[gap_bar:] *= (1.0 + gap_pct)
    return np.maximum(modified, prices[0] * 0.001)


# ---------------------------------------------------------------------------
# Scenario modifier functions (standalone, injectable)
# ---------------------------------------------------------------------------

def _modifier_crypto_winter(
    prices: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Crypto winter 2022: -80% over 300 bars, 3x vol."""
    return _apply_trend_modifier(prices, total_return=-1.609, rng=rng, noise_sigma=0.015)


def _modifier_flash_crash(
    prices: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Flash crash: -20% in 5 bars, recovery over 10 bars."""
    crash_bar = max(1, len(prices) // 4)
    return _apply_flash_crash(
        prices,
        crash_depth=math.log(0.80),  # -20%
        crash_bar=crash_bar,
        crash_bars=5,
        recovery_bars=10,
        rng=rng,
    )


def _modifier_persistent_chop(
    prices: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Persistent chop: pure OU, no BH formation possible."""
    return _apply_chop(prices, rng=rng, reversion_strength=50.0)


def _modifier_bh_false_signal(
    prices: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """BH mass reaches 1.8 via small returns then dissipates without breakout.

    Creates conditions for a false positive: accumulate timelike bars
    then inject a SPACELIKE bar that deflates mass before activation.
    """
    n = len(prices)
    modified = prices.copy()
    dt = DT_15M
    cf = DEFAULT_CF

    # Phase 1: constrained small returns to push bh_mass toward 1.8
    # We need many consecutive bars with |ret|/cf < 1
    # Use ret = cf * 0.5 (beta = 0.5, TIMELIKE)
    for i in range(1, min(100, n)):
        ret = cf * 0.5 * rng.choice([-1, 1]) * (0.9 + rng.random() * 0.2)
        modified[i] = modified[i - 1] * math.exp(float(ret))

    # Phase 2: SPACELIKE bar -- big return to decay mass
    spacelike_bar = min(100, n - 1)
    if spacelike_bar < n:
        modified[spacelike_bar] = modified[spacelike_bar - 1] * math.exp(cf * 3.0)

    # Phase 3: resume normal mild movement
    for i in range(spacelike_bar + 1, n):
        ret = cf * 0.4 * rng.standard_normal()
        modified[i] = modified[i - 1] * math.exp(float(ret))

    return np.maximum(modified, prices[0] * 0.001)


def _modifier_leverage_cascade(
    prices: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Leverage cascade: rapid vol expansion, forced liquidation gaps."""
    n = len(prices)
    # Series of gap-downs with escalating severity
    gaps = []
    gap_bar = max(10, n // 8)
    cumulative = 0.0
    for k in range(5):
        gap_size = -(0.03 + k * 0.02) * (1.0 + rng.uniform(-0.2, 0.2))
        cumulative += gap_size
        gaps.append((gap_bar + k * 8, gap_size))
    result = _apply_discontinuous_drop(prices, gaps, rng)
    # Add elevated vol throughout
    log_rets = np.diff(np.log(np.maximum(result, 1e-12)))
    log_rets *= (1.5 + rng.standard_normal(len(log_rets)) * 0.3)
    recon = np.concatenate([[result[0]], result[0] * np.exp(np.cumsum(log_rets))])
    return np.maximum(recon, prices[0] * 0.001)


def _modifier_covid_march(
    prices: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """COVID March 2020: -50% over 30 bars, VIX explosion, 200-bar recovery."""
    n = len(prices)
    crash_bar = max(1, n // 10)
    result = _apply_flash_crash(
        prices,
        crash_depth=math.log(0.50),  # -50%
        crash_bar=crash_bar,
        crash_bars=30,
        recovery_bars=200,
        rng=rng,
    )
    return result


def _modifier_fed_rate_shock(
    prices: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Fed rate shock: -15% gap open, sustained downtrend, regime flip."""
    n = len(prices)
    modified = prices.copy()
    gap_bar = max(1, n // 10)
    if gap_bar < n:
        # Instantaneous -15% gap
        modified[gap_bar:] *= 0.85
    # Sustained downtrend: additional -8% drift annualised
    post_n = n - gap_bar
    if post_n > 0:
        t = np.linspace(0, 1, post_n)
        trend = np.exp(-0.30 * t)  # ~-26% over full post-gap period
        modified[gap_bar:] *= trend
    return np.maximum(modified, prices[0] * 0.001)


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

STRESS_SCENARIOS: dict[str, StressScenario] = {
    "crypto_winter_2022": StressScenario(
        name="crypto_winter_2022",
        description=(
            "Prolonged bear market: -80% over 300 bars (approx 115 trading days), "
            "vol 3x normal, correlation to equities breaks down. "
            "Tests BH detection under sustained negative drift."
        ),
        price_path_modifier=_modifier_crypto_winter,
        vol_modifier=3.0,
        correlation_modifier=-0.5,
        duration_bars=300,
        initial_price=100.0,
        base_annual_vol=0.80,
        seed=42,
    ),
    "flash_crash": StressScenario(
        name="flash_crash",
        description=(
            "-20% drawdown in 5 bars followed by near-complete recovery in 10 bars. "
            "Tests stop-loss triggers, re-entry logic, and position sizing under "
            "sudden extreme price dislocation."
        ),
        price_path_modifier=_modifier_flash_crash,
        vol_modifier=2.5,
        correlation_modifier=0.2,
        duration_bars=200,
        initial_price=100.0,
        base_annual_vol=0.20,
        seed=42,
    ),
    "persistent_chop": StressScenario(
        name="persistent_chop",
        description=(
            "Extended mean-reverting regime with no directional moves large enough to "
            "trigger BH formation. Tests filter effectiveness -- strategy should "
            "produce near-zero BH activations and minimal PnL (not large losses)."
        ),
        price_path_modifier=_modifier_persistent_chop,
        vol_modifier=0.6,
        correlation_modifier=0.0,
        duration_bars=500,
        initial_price=100.0,
        base_annual_vol=0.15,
        seed=42,
    ),
    "bh_false_signal": StressScenario(
        name="bh_false_signal",
        description=(
            "BH mass accumulates to 1.8 via consecutive timelike bars, approaching "
            "BH_FORM=1.5, then a SPACELIKE bar deflates mass before bh_active fires. "
            "Tests that the strategy does not enter on near-misses."
        ),
        price_path_modifier=_modifier_bh_false_signal,
        vol_modifier=1.0,
        correlation_modifier=0.0,
        duration_bars=300,
        initial_price=100.0,
        base_annual_vol=0.20,
        seed=42,
    ),
    "leverage_cascade": StressScenario(
        name="leverage_cascade",
        description=(
            "Rapid vol expansion with a series of discontinuous gap-down events "
            "simulating forced liquidations. Price path has 5 gap events totalling "
            "-30% with elevated intraday vol throughout."
        ),
        price_path_modifier=_modifier_leverage_cascade,
        vol_modifier=4.0,
        correlation_modifier=0.4,
        duration_bars=400,
        initial_price=100.0,
        base_annual_vol=0.30,
        seed=42,
    ),
    "covid_march_2020": StressScenario(
        name="covid_march_2020",
        description=(
            "-50% over 30 bars replicating March 2020 selloff. VIX-like vol "
            "explosion (4x), followed by 200-bar recovery. Tests max drawdown "
            "handling and recovery detection logic."
        ),
        price_path_modifier=_modifier_covid_march,
        vol_modifier=4.0,
        correlation_modifier=0.5,
        duration_bars=500,
        initial_price=100.0,
        base_annual_vol=0.25,
        seed=42,
    ),
    "fed_rate_shock": StressScenario(
        name="fed_rate_shock",
        description=(
            "-15% overnight gap open followed by sustained downtrend. "
            "Regime flips from BULL to BEAR mid-scenario. Tests that strategy "
            "adapts direction and does not hold long positions through sustained decline."
        ),
        price_path_modifier=_modifier_fed_rate_shock,
        vol_modifier=1.8,
        correlation_modifier=0.1,
        duration_bars=400,
        initial_price=100.0,
        base_annual_vol=0.18,
        seed=42,
    ),
}


# ---------------------------------------------------------------------------
# Stress Tester
# ---------------------------------------------------------------------------

def _compute_drawdown(equity_curve: NDArray[np.float64]) -> tuple[float, int]:
    """Return (worst_drawdown, recovery_bars).

    worst_drawdown: negative float, max peak-to-trough decline
    recovery_bars: bars from trough to return to prior peak (-1 if not recovered)
    """
    n = len(equity_curve)
    if n < 2:
        return (0.0, 0)

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / (running_max + 1e-12)
    worst_dd = float(np.min(drawdown))
    trough_idx = int(np.argmin(drawdown))
    trough_val = equity_curve[trough_idx]
    peak_at_trough = running_max[trough_idx]

    # Find first bar after trough where price >= peak_at_trough
    recovery_bars = -1
    for i in range(trough_idx + 1, n):
        if equity_curve[i] >= peak_at_trough:
            recovery_bars = i - trough_idx
            break

    return (worst_dd, recovery_bars)


def _compute_sharpe(returns: NDArray[np.float64], periods_per_year: float = 252.0 * 6.5 * 4) -> float:
    """Annualised Sharpe ratio from per-bar returns."""
    if len(returns) < 2:
        return float("nan")
    mu = float(np.mean(returns))
    sigma = float(np.std(returns))
    if sigma < 1e-12:
        return float("nan")
    return float(mu / sigma * math.sqrt(periods_per_year))


def _count_bh_false_positives(
    closes: NDArray[np.float64],
    bh_active: NDArray[np.bool_],
    min_move_pct: float = 0.01,
    fwd_window: int = 10,
) -> int:
    """Count BH activations not followed by a directional move > min_move_pct.

    A BH activation is a bar where bh_active transitions from False to True.
    A false positive is one where the maximum abs return over the next
    fwd_window bars is less than min_move_pct.
    """
    n = len(closes)
    count = 0
    for i in range(1, n - 1):
        if bh_active[i] and not bh_active[i - 1]:
            # New BH activation
            end = min(i + fwd_window, n)
            fwd_rets = np.abs(closes[i:end] / closes[i] - 1.0)
            if len(fwd_rets) == 0 or float(np.max(fwd_rets)) < min_move_pct:
                count += 1
    return count


class StressTester:
    """
    Runs named stress scenarios against a strategy function.

    The strategy function signature:
        strategy_fn(bars: pd.DataFrame) -> np.ndarray
    Where bars is an OHLCV DataFrame with columns:
        open, high, low, close, volume, bh_mass, bh_active, bh_dir
    And the return is an array of per-bar positions (+1 long, -1 short, 0 flat).

    If strategy_fn is None, the tester measures market-level stats only.

    Usage
    -----
    >>> tester = StressTester(cf=0.0003)
    >>> result = tester.run_scenario("flash_crash", strategy_fn=my_fn)
    >>> tester.run_all_scenarios(strategy_fn=my_fn)
    """

    def __init__(
        self,
        cf: float = DEFAULT_CF,
        bh_form: float = BH_FORM_DEFAULT,
        seed: Optional[int] = None,
    ):
        self.cf = cf
        self.bh_form = bh_form
        self.rng = np.random.default_rng(seed)

    def _generate_scenario_bars(self, scenario: StressScenario) -> pd.DataFrame:
        """Generate OHLCV bars for a scenario, applying its price_path_modifier."""
        n = scenario.duration_bars
        annual_vol = scenario.base_annual_vol * scenario.vol_modifier

        # Base GBM path
        base_prices = GeometricBrownianMotion.generate(
            n,
            mu=0.0,
            sigma=scenario.base_annual_vol,
            dt=DT_15M,
            seed=scenario.seed,
            initial=scenario.initial_price,
        )

        # Apply scenario modifier
        rng_scen = np.random.default_rng(scenario.seed + 1000)
        modified_prices = scenario.price_path_modifier(base_prices, rng_scen)

        # Compute BH mass series
        bh_masses, bh_active = compute_bh_mass_series(
            modified_prices, cf=self.cf, bh_form=self.bh_form
        )

        # Build OHLCV bars
        sigma_bar = annual_vol * math.sqrt(DT_15M)
        records = []
        bh_dir_current = 0
        price_window: list[float] = []

        for i in range(n):
            c = float(modified_prices[i])
            bar_in_day = i % _BARS_PER_DAY
            noise = c * sigma_bar * 0.5
            rng_bar = np.random.default_rng(scenario.seed + i)
            o = c * math.exp(rng_bar.normal(0, sigma_bar * 0.4))
            h = max(o, c) + abs(rng_bar.normal(0, noise * 0.4))
            l = min(o, c) - abs(rng_bar.normal(0, noise * 0.4))
            vol_base = math.exp(math.log(max(c, 1.0)) + 5.0 + rng_bar.normal(0, 0.35))
            vol_base *= _intraday_volume_factor(bar_in_day)

            # BH direction
            if bh_active[i] and (i == 0 or not bh_active[i - 1]):
                lookback = min(20, len(price_window))
                if lookback > 0:
                    bh_dir_current = 1 if c > price_window[-lookback] else -1
                else:
                    bh_dir_current = 0
            elif not bh_active[i]:
                bh_dir_current = 0

            price_window.append(c)
            if len(price_window) > 50:
                price_window.pop(0)

            records.append({
                "open": max(o, 1e-6),
                "high": max(h, o, c),
                "low": min(l, o, c),
                "close": c,
                "volume": vol_base,
                "bh_mass": float(bh_masses[i]),
                "bh_active": bool(bh_active[i]),
                "bh_dir": bh_dir_current,
                "regime": "STRESS",
            })

        return pd.DataFrame(records)

    def run_scenario(
        self,
        scenario_name: str,
        strategy_fn: Optional[Callable[[pd.DataFrame], NDArray[np.float64]]] = None,
    ) -> StressResult:
        """Run a named stress scenario and return StressResult.

        Parameters
        ----------
        scenario_name:
            Key in STRESS_SCENARIOS dict.
        strategy_fn:
            Optional strategy function. See class docstring for signature.
            If None, only market-level stats are computed.

        Returns
        -------
        StressResult
        """
        if scenario_name not in STRESS_SCENARIOS:
            raise KeyError(
                f"Unknown scenario '{scenario_name}'. "
                f"Available: {list(STRESS_SCENARIOS.keys())}"
            )
        scenario = STRESS_SCENARIOS[scenario_name]
        bars = self._generate_scenario_bars(scenario)
        closes = bars["close"].values
        bh_active = bars["bh_active"].values.astype(bool)

        # Market-level drawdown
        market_dd, recovery_bars = _compute_drawdown(closes)
        signals_triggered = int(np.sum(bh_active))
        bh_fp = _count_bh_false_positives(closes, bh_active, fwd_window=10)

        # Strategy-level metrics
        strategy_pnl = 0.0
        sharpe = float("nan")
        max_pos = 0.0
        strategy_returns = np.array([])

        if strategy_fn is not None:
            try:
                positions = strategy_fn(bars)
                positions = np.asarray(positions, dtype=np.float64)
                if len(positions) == len(bars):
                    log_rets = np.concatenate([[0.0], np.diff(np.log(np.maximum(closes, 1e-12)))])
                    # Strategy return: position[t-1] * market_return[t]
                    strat_rets = np.zeros(len(bars))
                    strat_rets[1:] = positions[:-1] * log_rets[1:]
                    strategy_returns = strat_rets
                    equity = np.exp(np.cumsum(strat_rets))
                    strategy_pnl = float(equity[-1] - 1.0)
                    sharpe = _compute_sharpe(strat_rets)
                    max_pos = float(np.max(np.abs(positions)))
            except Exception as exc:
                logger.warning("strategy_fn raised exception in scenario %s: %s", scenario_name, exc)

        return StressResult(
            scenario_name=scenario_name,
            worst_drawdown=market_dd,
            recovery_bars=recovery_bars,
            strategy_pnl=strategy_pnl,
            max_position_held=max_pos,
            signals_triggered=signals_triggered,
            sharpe_ratio=sharpe,
            bh_false_positives=bh_fp,
            price_path=closes,
            strategy_returns=strategy_returns,
        )

    def run_all_scenarios(
        self,
        strategy_fn: Optional[Callable[[pd.DataFrame], NDArray[np.float64]]] = None,
    ) -> dict[str, StressResult]:
        """Run all registered stress scenarios.

        Returns
        -------
        dict mapping scenario_name -> StressResult
        """
        results: dict[str, StressResult] = {}
        for name in STRESS_SCENARIOS:
            logger.info("Running stress scenario: %s", name)
            try:
                results[name] = self.run_scenario(name, strategy_fn=strategy_fn)
            except Exception as exc:
                logger.error("Scenario %s failed: %s", name, exc)
        return results

    def summary_table(self, results: dict[str, StressResult]) -> pd.DataFrame:
        """Build a summary DataFrame from a dict of StressResult objects.

        Columns: scenario, worst_drawdown, recovery_bars, strategy_pnl,
                 sharpe_ratio, signals_triggered, bh_false_positives
        """
        rows = []
        for name, r in results.items():
            rows.append({
                "scenario": name,
                "worst_drawdown": round(r.worst_drawdown, 4),
                "recovery_bars": r.recovery_bars,
                "strategy_pnl": round(r.strategy_pnl, 4),
                "sharpe_ratio": round(r.sharpe_ratio, 3) if not math.isnan(r.sharpe_ratio) else float("nan"),
                "signals_triggered": r.signals_triggered,
                "bh_false_positives": r.bh_false_positives,
            })
        return pd.DataFrame(rows).set_index("scenario")

    def assert_scenario_bounds(
        self,
        result: StressResult,
        max_drawdown: float = -0.50,
        min_recovery_bars: int = 1,
        max_false_positives: int = 5,
    ) -> list[str]:
        """Check whether a result violates specified risk bounds.

        Returns list of violation strings (empty = pass).
        """
        violations: list[str] = []
        if result.worst_drawdown < max_drawdown:
            violations.append(
                f"worst_drawdown {result.worst_drawdown:.3f} < threshold {max_drawdown:.3f}"
            )
        if result.bh_false_positives > max_false_positives:
            violations.append(
                f"bh_false_positives {result.bh_false_positives} > threshold {max_false_positives}"
            )
        return violations
