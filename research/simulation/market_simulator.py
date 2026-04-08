"""
market_simulator.py -- Full multi-asset market simulator with regime switching,
order book dynamics, market impact, flash crashes, event-driven gaps, and
intraday microstructure patterns.

All heavy numerics use numpy/scipy.  No external market-data dependencies.
"""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as sla
from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# ===================================================================
# 1.  Correlation / Cholesky helpers
# ===================================================================

def build_correlation_matrix(n_assets: int, rho: float = 0.5) -> FloatArray:
    """Equi-correlation matrix with off-diag = rho."""
    C = np.full((n_assets, n_assets), rho)
    np.fill_diagonal(C, 1.0)
    return C


def nearest_psd(mat: FloatArray, eps: float = 1e-8) -> FloatArray:
    """Project a symmetric matrix to the nearest positive-semi-definite matrix
    using the Higham alternating projection algorithm (simplified)."""
    B = (mat + mat.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def cholesky_lower(corr: FloatArray) -> FloatArray:
    """Return lower-triangular Cholesky factor, projecting to PSD first."""
    psd = nearest_psd(corr)
    return np.linalg.cholesky(psd)


def correlated_normals(
    n_steps: int, n_assets: int, corr: FloatArray, rng: np.random.Generator
) -> FloatArray:
    """Generate (n_steps, n_assets) correlated standard normals."""
    L = cholesky_lower(corr)
    Z = rng.standard_normal((n_steps, n_assets))
    return Z @ L.T


# ===================================================================
# 2.  Heston stochastic-volatility engine
# ===================================================================

@dataclass
class HestonParams:
    """Per-asset Heston parameters."""
    v0: float = 0.04          # initial variance
    kappa: float = 2.0        # mean-reversion speed
    theta: float = 0.04       # long-run variance
    xi: float = 0.3           # vol of vol
    rho_sv: float = -0.7      # spot-vol correlation


def heston_step(
    v: FloatArray,
    params: Sequence[HestonParams],
    dt: float,
    Z_v: FloatArray,
) -> FloatArray:
    """Euler-step for CIR variance process (one step, all assets)."""
    n = len(params)
    kappa = np.array([p.kappa for p in params])
    theta = np.array([p.theta for p in params])
    xi = np.array([p.xi for p in params])
    v_pos = np.maximum(v, 0.0)
    dv = kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * Z_v
    return np.maximum(v + dv, 0.0)


def heston_qe_step(
    v: FloatArray,
    params: Sequence[HestonParams],
    dt: float,
    U: FloatArray,
) -> FloatArray:
    """Quadratic-Exponential (QE) scheme for CIR variance -- per-asset."""
    n = len(params)
    kappa = np.array([p.kappa for p in params])
    theta = np.array([p.theta for p in params])
    xi = np.array([p.xi for p in params])
    v_pos = np.maximum(v, 1e-12)

    exp_k = np.exp(-kappa * dt)
    m = theta + (v_pos - theta) * exp_k
    s2 = (
        v_pos * xi ** 2 * exp_k / kappa * (1.0 - exp_k)
        + theta * xi ** 2 / (2.0 * kappa) * (1.0 - exp_k) ** 2
    )
    psi = s2 / (m ** 2 + 1e-30)

    v_new = np.empty_like(v)
    # Region psi <= 1.5  -- quadratic
    mask_q = psi <= 1.5
    b2 = np.where(mask_q, 2.0 / psi - 1.0 + np.sqrt(2.0 / psi) * np.sqrt(np.maximum(2.0 / psi - 1.0, 0.0)), 0.0)
    a_coeff = np.where(mask_q, m / (1.0 + b2), 0.0)
    Zv = stats.norm.ppf(np.clip(U, 1e-12, 1.0 - 1e-12))
    v_new = np.where(mask_q, a_coeff * (np.sqrt(b2) + Zv) ** 2, v_new)

    # Region psi > 1.5  -- exponential
    mask_e = ~mask_q
    p_exp = (psi - 1.0) / (psi + 1e-30)
    beta_e = np.where(mask_e, (1.0 - p_exp) / (m + 1e-30), 0.0)
    v_exp = np.where(
        U <= p_exp,
        0.0,
        np.where(mask_e, np.log((1.0 - p_exp) / (1.0 - U + 1e-30)) / (beta_e + 1e-30), 0.0),
    )
    v_new = np.where(mask_e, v_exp, v_new)
    return np.maximum(v_new, 0.0)


# ===================================================================
# 3.  Jump-diffusion (Merton)
# ===================================================================

@dataclass
class MertonJumpParams:
    lam: float = 0.1          # jump intensity (per year)
    mu_j: float = -0.05       # mean log jump
    sigma_j: float = 0.10     # std  log jump


def merton_jump_component(
    n_steps: int,
    n_assets: int,
    params: Sequence[MertonJumpParams],
    dt: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Return (n_steps, n_assets) of multiplicative jump factors."""
    jumps = np.ones((n_steps, n_assets))
    for j, p in enumerate(params):
        N_j = rng.poisson(p.lam * dt, size=n_steps)
        for i in range(n_steps):
            if N_j[i] > 0:
                J = np.sum(rng.normal(p.mu_j, p.sigma_j, size=N_j[i]))
                jumps[i, j] = np.exp(J)
    return jumps


# ===================================================================
# 4.  Regime switching
# ===================================================================

class MarketRegime(enum.IntEnum):
    BULL = 0
    BEAR = 1
    CRISIS = 2
    RECOVERY = 3


@dataclass
class RegimeParams:
    drift: FloatArray          # (n_assets,)
    vol: FloatArray            # (n_assets,)
    corr: FloatArray           # (n_assets, n_assets)


@dataclass
class RegimeSwitchConfig:
    """Markov transition matrix + per-regime params."""
    transition_matrix: FloatArray          # (n_regimes, n_regimes)
    regime_params: Dict[int, RegimeParams] = field(default_factory=dict)

    def validate(self) -> None:
        row_sums = self.transition_matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0), f"Row sums {row_sums} != 1"

    def sample_next_regime(self, current: int, rng: np.random.Generator) -> int:
        probs = self.transition_matrix[current]
        return int(rng.choice(len(probs), p=probs))


def default_regime_switch(n_assets: int) -> RegimeSwitchConfig:
    """4-regime default config."""
    T = np.array([
        [0.95, 0.03, 0.01, 0.01],
        [0.05, 0.90, 0.04, 0.01],
        [0.02, 0.03, 0.85, 0.10],
        [0.10, 0.02, 0.03, 0.85],
    ])
    base_drift = np.full(n_assets, 0.10)
    base_vol = np.full(n_assets, 0.20)
    base_corr = build_correlation_matrix(n_assets, 0.3)
    params = {
        MarketRegime.BULL: RegimeParams(base_drift * 1.5, base_vol * 0.8, build_correlation_matrix(n_assets, 0.25)),
        MarketRegime.BEAR: RegimeParams(-base_drift * 0.5, base_vol * 1.3, build_correlation_matrix(n_assets, 0.50)),
        MarketRegime.CRISIS: RegimeParams(-base_drift * 2.0, base_vol * 2.5, build_correlation_matrix(n_assets, 0.80)),
        MarketRegime.RECOVERY: RegimeParams(base_drift * 1.0, base_vol * 1.0, build_correlation_matrix(n_assets, 0.35)),
    }
    cfg = RegimeSwitchConfig(transition_matrix=T, regime_params=params)
    cfg.validate()
    return cfg


def simulate_regime_path(
    n_steps: int,
    config: RegimeSwitchConfig,
    initial_regime: int = 0,
    rng: np.random.Generator | None = None,
) -> IntArray:
    rng = rng or np.random.default_rng()
    regimes = np.empty(n_steps, dtype=np.int64)
    regimes[0] = initial_regime
    for t in range(1, n_steps):
        regimes[t] = config.sample_next_regime(int(regimes[t - 1]), rng)
    return regimes


# ===================================================================
# 5.  Order book simulation
# ===================================================================

@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBookSnapshot:
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: float = 0.0

    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2.0
        return 0.0

    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return float("inf")

    @property
    def bid_depth(self) -> float:
        return sum(l.size for l in self.bids)

    @property
    def ask_depth(self) -> float:
        return sum(l.size for l in self.asks)


class OrderBookSimulator:
    """Poisson limit-order arrivals with exponential sizes and cancellations."""

    def __init__(
        self,
        mid_price: float = 100.0,
        tick_size: float = 0.01,
        n_levels: int = 20,
        arrival_rate: float = 50.0,
        cancel_rate: float = 30.0,
        mean_size: float = 100.0,
        spread_ticks: int = 2,
        rng: np.random.Generator | None = None,
    ):
        self.tick = tick_size
        self.n_levels = n_levels
        self.arrival_rate = arrival_rate
        self.cancel_rate = cancel_rate
        self.mean_size = mean_size
        self.spread_ticks = spread_ticks
        self.rng = rng or np.random.default_rng()
        self._init_book(mid_price)

    def _init_book(self, mid: float) -> None:
        self.bids: List[OrderBookLevel] = []
        self.asks: List[OrderBookLevel] = []
        half = self.spread_ticks * self.tick / 2.0
        for i in range(self.n_levels):
            bp = mid - half - i * self.tick
            ap = mid + half + i * self.tick
            bs = self.rng.exponential(self.mean_size)
            asz = self.rng.exponential(self.mean_size)
            self.bids.append(OrderBookLevel(round(bp, 8), bs))
            self.asks.append(OrderBookLevel(round(ap, 8), asz))

    def step(self, dt: float) -> OrderBookSnapshot:
        n_arrivals = self.rng.poisson(self.arrival_rate * dt)
        for _ in range(n_arrivals):
            side = self.rng.choice(2)
            size = self.rng.exponential(self.mean_size)
            level_idx = min(int(self.rng.exponential(3.0)), self.n_levels - 1)
            book = self.bids if side == 0 else self.asks
            if level_idx < len(book):
                book[level_idx] = OrderBookLevel(
                    book[level_idx].price, book[level_idx].size + size
                )

        n_cancels = self.rng.poisson(self.cancel_rate * dt)
        for _ in range(n_cancels):
            side = self.rng.choice(2)
            book = self.bids if side == 0 else self.asks
            if len(book) == 0:
                continue
            level_idx = min(int(self.rng.exponential(2.0)), len(book) - 1)
            cancel_size = self.rng.exponential(self.mean_size * 0.5)
            new_size = max(book[level_idx].size - cancel_size, 1.0)
            book[level_idx] = OrderBookLevel(book[level_idx].price, new_size)

        return OrderBookSnapshot(
            bids=list(self.bids), asks=list(self.asks), timestamp=0.0
        )

    def apply_market_order(self, side: int, quantity: float) -> float:
        """Execute a market order.  side=0 buy (lifts asks), side=1 sell (hits bids).
        Returns average fill price."""
        book = self.asks if side == 0 else self.bids
        filled = 0.0
        cost = 0.0
        remaining = quantity
        while remaining > 0 and book:
            level = book[0]
            fill = min(remaining, level.size)
            cost += fill * level.price
            filled += fill
            remaining -= fill
            if fill >= level.size:
                book.pop(0)
            else:
                book[0] = OrderBookLevel(level.price, level.size - fill)
        return cost / max(filled, 1e-12)

    def snapshot(self, t: float = 0.0) -> OrderBookSnapshot:
        return OrderBookSnapshot(
            bids=list(self.bids), asks=list(self.asks), timestamp=t
        )


# ===================================================================
# 6.  Market impact models
# ===================================================================

@dataclass
class ImpactParams:
    permanent_lambda: float = 1e-5       # Kyle lambda
    temporary_eta: float = 1e-4          # temporary cost coefficient
    sqrt_gamma: float = 0.5             # exponent for sqrt model
    decay_half_life: float = 10.0       # temporary impact decay in steps


class MarketImpactModel:
    """Combined permanent + temporary impact with square-root model."""

    def __init__(self, params: ImpactParams | None = None):
        self.params = params or ImpactParams()
        self._cumulative_flow = 0.0
        self._temp_impacts: List[Tuple[float, float]] = []  # (time, impact)

    def reset(self) -> None:
        self._cumulative_flow = 0.0
        self._temp_impacts.clear()

    def permanent_impact(self, order_flow: float) -> float:
        self._cumulative_flow += order_flow
        return self.params.permanent_lambda * self._cumulative_flow

    def temporary_impact(self, order_flow: float, adv: float, t: float) -> float:
        sigma_est = 0.02
        participation = abs(order_flow) / max(adv, 1.0)
        sign = np.sign(order_flow)
        impact = (
            sign
            * self.params.temporary_eta
            * sigma_est
            * (participation ** self.params.sqrt_gamma)
        )
        self._temp_impacts.append((t, impact))
        return impact

    def total_impact(
        self, order_flow: float, adv: float, t: float
    ) -> float:
        perm = self.permanent_impact(order_flow)
        temp = self.temporary_impact(order_flow, adv, t)
        decayed = self._decayed_temp(t)
        return perm + decayed

    def _decayed_temp(self, t: float) -> float:
        hl = self.params.decay_half_life
        total = 0.0
        alive = []
        for t0, imp in self._temp_impacts:
            decay = np.exp(-np.log(2) * (t - t0) / hl)
            if decay > 0.01:
                total += imp * decay
                alive.append((t0, imp))
        self._temp_impacts = alive
        return total

    def kyle_lambda_estimate(
        self, prices: FloatArray, volumes: FloatArray
    ) -> float:
        """Regress |delta_p| on signed sqrt volume."""
        dp = np.diff(prices)
        signed_vol = np.sign(dp) * np.sqrt(np.abs(volumes[1:]))
        if len(dp) < 5:
            return self.params.permanent_lambda
        X = signed_vol.reshape(-1, 1)
        beta = np.linalg.lstsq(X, dp, rcond=None)[0]
        return float(abs(beta[0]))


# ===================================================================
# 7.  Flash crash model
# ===================================================================

@dataclass
class FlashCrashConfig:
    trigger_prob: float = 0.001          # per-step probability of flash crash
    depth_withdrawal_pct: float = 0.80   # fraction of liquidity removed
    cascade_steps: int = 10              # number of cascade steps
    recovery_steps: int = 50             # steps to recover
    max_drop_pct: float = 0.08           # maximum drop from pre-crash


class FlashCrashEngine:
    """Simulates sudden liquidity withdrawal and price cascade."""

    def __init__(self, config: FlashCrashConfig | None = None, rng: np.random.Generator | None = None):
        self.cfg = config or FlashCrashConfig()
        self.rng = rng or np.random.default_rng()
        self._in_crash = False
        self._crash_step = 0
        self._pre_crash_price = 0.0
        self._crash_bottom = 0.0

    def check_trigger(self, t: int) -> bool:
        if self._in_crash:
            return False
        return bool(self.rng.random() < self.cfg.trigger_prob)

    def start_crash(self, current_price: float) -> None:
        self._in_crash = True
        self._crash_step = 0
        self._pre_crash_price = current_price
        drop = self.rng.uniform(0.03, self.cfg.max_drop_pct)
        self._crash_bottom = current_price * (1.0 - drop)

    def crash_price_adjustment(self, current_price: float) -> float:
        if not self._in_crash:
            return 0.0
        self._crash_step += 1
        if self._crash_step <= self.cfg.cascade_steps:
            progress = self._crash_step / self.cfg.cascade_steps
            target = self._pre_crash_price + progress * (
                self._crash_bottom - self._pre_crash_price
            )
            return target - current_price
        elif self._crash_step <= self.cfg.cascade_steps + self.cfg.recovery_steps:
            rec_progress = (self._crash_step - self.cfg.cascade_steps) / self.cfg.recovery_steps
            target = self._crash_bottom + rec_progress * (
                self._pre_crash_price - self._crash_bottom
            ) * 0.9
            adj = (target - current_price) * 0.1
            return adj
        else:
            self._in_crash = False
            return 0.0

    def modify_order_book(self, book: OrderBookSimulator) -> None:
        if not self._in_crash:
            return
        if self._crash_step <= self.cfg.cascade_steps:
            for i in range(len(book.bids)):
                book.bids[i] = OrderBookLevel(
                    book.bids[i].price,
                    book.bids[i].size * (1.0 - self.cfg.depth_withdrawal_pct),
                )


# ===================================================================
# 8.  Earnings / event model
# ===================================================================

@dataclass
class EventSchedule:
    event_times: List[int]               # step indices
    gap_mean: float = 0.02               # mean absolute gap
    gap_std: float = 0.01
    vol_spike_mult: float = 3.0          # vol multiplier at event
    vol_decay_steps: int = 5


class EventEngine:
    """Scheduled gaps (earnings, FOMC) and vol spikes."""

    def __init__(self, schedule: EventSchedule | None = None, rng: np.random.Generator | None = None):
        self.schedule = schedule or EventSchedule(event_times=[])
        self.rng = rng or np.random.default_rng()
        self._vol_boost: Dict[int, float] = {}

    def add_events_random(self, n_steps: int, n_events: int = 4) -> None:
        times = sorted(self.rng.choice(n_steps, size=n_events, replace=False).tolist())
        self.schedule.event_times = times

    def gap_at(self, t: int) -> float:
        if t in self.schedule.event_times:
            sign = self.rng.choice([-1, 1])
            mag = abs(self.rng.normal(self.schedule.gap_mean, self.schedule.gap_std))
            return sign * mag
        return 0.0

    def vol_multiplier_at(self, t: int) -> float:
        mult = 1.0
        for ev_t in self.schedule.event_times:
            if ev_t <= t < ev_t + self.schedule.vol_decay_steps:
                decay = (t - ev_t) / self.schedule.vol_decay_steps
                mult = max(mult, 1.0 + (self.schedule.vol_spike_mult - 1.0) * (1.0 - decay))
        return mult


# ===================================================================
# 9.  Intraday patterns
# ===================================================================

@dataclass
class IntradayPatternConfig:
    trading_minutes: int = 390           # 6.5 hours
    open_vol_mult: float = 2.0
    close_vol_mult: float = 1.5
    midday_vol_mult: float = 0.7
    open_spread_mult: float = 2.5
    close_spread_mult: float = 1.8


class IntradayPatternEngine:
    """U-shaped volume curve and spread widening at open/close."""

    def __init__(self, config: IntradayPatternConfig | None = None):
        self.cfg = config or IntradayPatternConfig()
        self._build_curves()

    def _build_curves(self) -> None:
        T = self.cfg.trading_minutes
        x = np.array([0, T * 0.05, T * 0.15, T * 0.5, T * 0.85, T * 0.95, T])
        vol_y = np.array([
            self.cfg.open_vol_mult,
            self.cfg.open_vol_mult * 0.8,
            1.0,
            self.cfg.midday_vol_mult,
            1.0,
            self.cfg.close_vol_mult * 0.8,
            self.cfg.close_vol_mult,
        ])
        spread_y = np.array([
            self.cfg.open_spread_mult,
            1.5,
            1.0,
            1.0,
            1.0,
            1.3,
            self.cfg.close_spread_mult,
        ])
        self._vol_curve = CubicSpline(x, vol_y)
        self._spread_curve = CubicSpline(x, spread_y)

    def volume_multiplier(self, minute_of_day: float) -> float:
        return float(np.clip(self._vol_curve(minute_of_day), 0.3, 5.0))

    def spread_multiplier(self, minute_of_day: float) -> float:
        return float(np.clip(self._spread_curve(minute_of_day), 0.5, 5.0))


# ===================================================================
# 10. Central bank / rate shock model
# ===================================================================

@dataclass
class CentralBankConfig:
    shock_times: List[int] = field(default_factory=list)
    rate_change_bps: List[float] = field(default_factory=list)
    equity_sensitivity: float = -5.0     # pct move per 100bp rate surprise
    vol_spike_mult: float = 2.0
    vol_decay_steps: int = 20


class CentralBankEngine:
    """Rate shock events with equity and vol response."""

    def __init__(self, config: CentralBankConfig | None = None):
        self.cfg = config or CentralBankConfig()

    def price_shock_at(self, t: int) -> float:
        if t in self.cfg.shock_times:
            idx = self.cfg.shock_times.index(t)
            bps = self.cfg.rate_change_bps[idx] if idx < len(self.cfg.rate_change_bps) else 0.0
            return self.cfg.equity_sensitivity * bps / 10000.0
        return 0.0

    def vol_multiplier_at(self, t: int) -> float:
        mult = 1.0
        for st in self.cfg.shock_times:
            if st <= t < st + self.cfg.vol_decay_steps:
                decay = (t - st) / self.cfg.vol_decay_steps
                mult = max(mult, 1.0 + (self.cfg.vol_spike_mult - 1.0) * (1.0 - decay))
        return mult


# ===================================================================
# 11. Simulation output containers
# ===================================================================

@dataclass
class SimulationOutput:
    """Full output of a market simulation run."""
    prices: FloatArray                    # (n_steps, n_assets)
    volumes: FloatArray                   # (n_steps, n_assets)
    variances: FloatArray                 # (n_steps, n_assets) -- Heston variance
    spreads: FloatArray                   # (n_steps,)
    regimes: IntArray                     # (n_steps,)
    order_book_snapshots: List[OrderBookSnapshot]
    impact_costs: FloatArray              # (n_steps,)
    flash_crash_mask: FloatArray          # (n_steps,) bool
    event_mask: FloatArray                # (n_steps,) bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def log_returns(self) -> FloatArray:
        return np.diff(np.log(self.prices + 1e-30), axis=0)

    @property
    def simple_returns(self) -> FloatArray:
        return np.diff(self.prices, axis=0) / (self.prices[:-1] + 1e-30)

    def realized_volatility(self, window: int = 21) -> FloatArray:
        lr = self.log_returns
        n, m = lr.shape
        rv = np.full((n, m), np.nan)
        for i in range(window, n):
            rv[i] = np.std(lr[i - window : i], axis=0) * np.sqrt(252)
        return rv

    def sharpe_ratio(self, annualization: float = 252.0) -> FloatArray:
        lr = self.log_returns
        return lr.mean(axis=0) / (lr.std(axis=0) + 1e-12) * np.sqrt(annualization)

    def max_drawdown(self) -> FloatArray:
        cum = np.cumprod(1.0 + self.simple_returns, axis=0)
        running_max = np.maximum.accumulate(cum, axis=0)
        dd = (cum - running_max) / (running_max + 1e-12)
        return dd.min(axis=0)

    def correlation_matrix(self, window: int | None = None) -> FloatArray:
        lr = self.log_returns
        if window is not None:
            lr = lr[-window:]
        return np.corrcoef(lr.T)


# ===================================================================
# 12. Master simulator
# ===================================================================

@dataclass
class SimulatorConfig:
    n_assets: int = 5
    n_steps: int = 252 * 10
    dt: float = 1.0 / 252.0
    initial_prices: FloatArray | None = None
    drift: FloatArray | None = None
    correlation: FloatArray | None = None
    heston_params: List[HestonParams] | None = None
    merton_params: List[MertonJumpParams] | None = None
    regime_config: RegimeSwitchConfig | None = None
    use_heston: bool = True
    use_jumps: bool = True
    use_regime_switching: bool = True
    use_order_book: bool = False
    use_impact: bool = False
    use_flash_crash: bool = True
    use_events: bool = True
    use_intraday: bool = False
    use_central_bank: bool = False
    flash_crash_config: FlashCrashConfig | None = None
    event_schedule: EventSchedule | None = None
    intraday_config: IntradayPatternConfig | None = None
    central_bank_config: CentralBankConfig | None = None
    impact_params: ImpactParams | None = None
    seed: int = 42


class MarketSimulator:
    """Full multi-asset market simulator combining all engines."""

    def __init__(self, config: SimulatorConfig | None = None):
        self.cfg = config or SimulatorConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._setup_defaults()
        self._setup_engines()

    def _setup_defaults(self) -> None:
        n = self.cfg.n_assets
        if self.cfg.initial_prices is None:
            self.cfg.initial_prices = np.full(n, 100.0)
        if self.cfg.drift is None:
            self.cfg.drift = np.full(n, 0.08)
        if self.cfg.correlation is None:
            self.cfg.correlation = build_correlation_matrix(n, 0.4)
        if self.cfg.heston_params is None:
            self.cfg.heston_params = [HestonParams() for _ in range(n)]
        if self.cfg.merton_params is None:
            self.cfg.merton_params = [MertonJumpParams() for _ in range(n)]
        if self.cfg.regime_config is None and self.cfg.use_regime_switching:
            self.cfg.regime_config = default_regime_switch(n)

    def _setup_engines(self) -> None:
        self.flash_engine = FlashCrashEngine(self.cfg.flash_crash_config, self.rng)
        self.event_engine = EventEngine(self.cfg.event_schedule, self.rng)
        if self.cfg.use_events and not self.event_engine.schedule.event_times:
            self.event_engine.add_events_random(self.cfg.n_steps, n_events=8)
        self.intraday_engine = IntradayPatternEngine(self.cfg.intraday_config)
        self.cb_engine = CentralBankEngine(self.cfg.central_bank_config)
        self.impact_model = MarketImpactModel(self.cfg.impact_params)
        self.ob_sim: OrderBookSimulator | None = None
        if self.cfg.use_order_book:
            self.ob_sim = OrderBookSimulator(
                mid_price=float(self.cfg.initial_prices[0]),
                rng=self.rng,
            )

    def run(self) -> SimulationOutput:
        """Execute the full simulation."""
        n_steps = self.cfg.n_steps
        n_assets = self.cfg.n_assets
        dt = self.cfg.dt

        prices = np.zeros((n_steps, n_assets))
        volumes = np.zeros((n_steps, n_assets))
        variances = np.zeros((n_steps, n_assets))
        spreads = np.zeros(n_steps)
        impact_costs = np.zeros(n_steps)
        flash_mask = np.zeros(n_steps, dtype=bool)
        event_mask = np.zeros(n_steps, dtype=bool)
        ob_snaps: List[OrderBookSnapshot] = []

        prices[0] = self.cfg.initial_prices
        v0 = np.array([p.v0 for p in self.cfg.heston_params])
        variances[0] = v0

        # Regime path
        if self.cfg.use_regime_switching:
            regimes = simulate_regime_path(n_steps, self.cfg.regime_config, rng=self.rng)
        else:
            regimes = np.zeros(n_steps, dtype=np.int64)

        # Pre-generate correlated normals for base case
        Z_all = correlated_normals(n_steps, n_assets, self.cfg.correlation, self.rng)
        Z_vol = self.rng.standard_normal((n_steps, n_assets))
        U_qe = self.rng.random((n_steps, n_assets))

        # Jump components
        if self.cfg.use_jumps:
            jump_factors = merton_jump_component(
                n_steps, n_assets, self.cfg.merton_params, dt, self.rng
            )
        else:
            jump_factors = np.ones((n_steps, n_assets))

        current_v = v0.copy()

        for t in range(1, n_steps):
            regime = int(regimes[t])

            # Per-regime drift/vol
            if self.cfg.use_regime_switching and regime in self.cfg.regime_config.regime_params:
                rp = self.cfg.regime_config.regime_params[regime]
                mu_t = rp.drift
                base_vol_t = rp.vol
            else:
                mu_t = self.cfg.drift
                base_vol_t = np.full(n_assets, 0.20)

            # Heston variance step
            if self.cfg.use_heston:
                current_v = heston_qe_step(current_v, self.cfg.heston_params, dt, U_qe[t])
                sigma_t = np.sqrt(np.maximum(current_v, 1e-12))
            else:
                sigma_t = base_vol_t

            # Event vol multiplier
            vol_mult = 1.0
            if self.cfg.use_events:
                vol_mult *= self.event_engine.vol_multiplier_at(t)
                gap = self.event_engine.gap_at(t)
                if gap != 0.0:
                    event_mask[t] = True
            else:
                gap = 0.0

            if self.cfg.use_central_bank:
                vol_mult *= self.cb_engine.vol_multiplier_at(t)
                cb_shock = self.cb_engine.price_shock_at(t)
            else:
                cb_shock = 0.0

            sigma_t = sigma_t * vol_mult

            # GBM step
            dW = Z_all[t] * np.sqrt(dt)
            log_ret = (mu_t - 0.5 * sigma_t ** 2) * dt + sigma_t * dW

            # Add gap + cb shock
            log_ret += gap + cb_shock

            # Multiply by jump factor
            new_prices = prices[t - 1] * np.exp(log_ret) * jump_factors[t]

            # Flash crash
            if self.cfg.use_flash_crash:
                if self.flash_engine.check_trigger(t):
                    self.flash_engine.start_crash(float(new_prices[0]))
                adj = self.flash_engine.crash_price_adjustment(float(new_prices[0]))
                if adj != 0.0:
                    flash_mask[t] = True
                    new_prices[0] += adj
                    # Contagion: other assets move partially
                    new_prices[1:] += adj * 0.5 * self.rng.uniform(0.3, 0.8, size=n_assets - 1)

            new_prices = np.maximum(new_prices, 1e-6)
            prices[t] = new_prices
            variances[t] = current_v

            # Volume: base proportional to vol, with random noise
            base_vol_est = sigma_t * np.sqrt(dt)
            volumes[t] = (
                1e6
                * base_vol_est
                * self.rng.lognormal(0, 0.3, size=n_assets)
            )

            # Intraday pattern
            if self.cfg.use_intraday:
                minute = (t % 390)
                volumes[t] *= self.intraday_engine.volume_multiplier(minute)
                sp_mult = self.intraday_engine.spread_multiplier(minute)
            else:
                sp_mult = 1.0

            # Spread estimate
            base_spread = 0.01 * (sigma_t[0] / 0.2) * sp_mult
            spreads[t] = base_spread

            # Order book step
            if self.cfg.use_order_book and self.ob_sim is not None:
                snap = self.ob_sim.step(dt)
                snap.timestamp = t * dt
                if self.cfg.use_flash_crash and flash_mask[t]:
                    self.flash_engine.modify_order_book(self.ob_sim)
                ob_snaps.append(snap)

            # Impact
            if self.cfg.use_impact:
                flow = self.rng.normal(0, volumes[t, 0] * 0.01)
                ic = self.impact_model.total_impact(flow, volumes[t, 0], float(t))
                impact_costs[t] = abs(ic)

        return SimulationOutput(
            prices=prices,
            volumes=volumes,
            variances=variances,
            spreads=spreads,
            regimes=regimes,
            order_book_snapshots=ob_snaps,
            impact_costs=impact_costs,
            flash_crash_mask=flash_mask.astype(np.float64),
            event_mask=event_mask.astype(np.float64),
            metadata={
                "config_n_assets": n_assets,
                "config_n_steps": n_steps,
                "config_dt": dt,
                "seed": self.cfg.seed,
            },
        )


# ===================================================================
# 13. Scenario generators
# ===================================================================

def generate_bull_market(n_assets: int = 5, n_years: int = 5) -> SimulationOutput:
    """Pre-configured bull market scenario."""
    cfg = SimulatorConfig(
        n_assets=n_assets,
        n_steps=252 * n_years,
        drift=np.full(n_assets, 0.15),
        use_regime_switching=False,
        use_flash_crash=False,
        use_events=False,
    )
    return MarketSimulator(cfg).run()


def generate_crisis_scenario(n_assets: int = 5) -> SimulationOutput:
    """2008-style crisis scenario."""
    cfg = SimulatorConfig(
        n_assets=n_assets,
        n_steps=504,
        drift=np.full(n_assets, -0.30),
        correlation=build_correlation_matrix(n_assets, 0.85),
        heston_params=[HestonParams(v0=0.09, theta=0.12, xi=0.8, kappa=1.0) for _ in range(n_assets)],
        merton_params=[MertonJumpParams(lam=0.3, mu_j=-0.08, sigma_j=0.15) for _ in range(n_assets)],
        use_regime_switching=False,
        use_flash_crash=True,
        flash_crash_config=FlashCrashConfig(trigger_prob=0.01, max_drop_pct=0.12),
    )
    return MarketSimulator(cfg).run()


def generate_low_vol_grind(n_assets: int = 5) -> SimulationOutput:
    """Low volatility, steady uptrend."""
    cfg = SimulatorConfig(
        n_assets=n_assets,
        n_steps=756,
        drift=np.full(n_assets, 0.10),
        heston_params=[HestonParams(v0=0.01, theta=0.01, xi=0.1) for _ in range(n_assets)],
        use_jumps=False,
        use_flash_crash=False,
        use_events=False,
        use_regime_switching=False,
    )
    return MarketSimulator(cfg).run()


def generate_choppy_sideways(n_assets: int = 5) -> SimulationOutput:
    """High vol, zero drift."""
    cfg = SimulatorConfig(
        n_assets=n_assets,
        n_steps=504,
        drift=np.zeros(n_assets),
        heston_params=[HestonParams(v0=0.06, theta=0.06, xi=0.5) for _ in range(n_assets)],
        use_regime_switching=False,
    )
    return MarketSimulator(cfg).run()


def generate_rate_hike_scenario(n_assets: int = 5) -> SimulationOutput:
    """Central bank tightening with multiple rate hikes."""
    cb_cfg = CentralBankConfig(
        shock_times=[63, 126, 189, 252],
        rate_change_bps=[25.0, 50.0, 75.0, 50.0],
        equity_sensitivity=-6.0,
    )
    cfg = SimulatorConfig(
        n_assets=n_assets,
        n_steps=504,
        use_central_bank=True,
        central_bank_config=cb_cfg,
    )
    return MarketSimulator(cfg).run()


# ===================================================================
# 14. Batch simulation & Monte Carlo
# ===================================================================

class MonteCarloSimulator:
    """Run many paths and aggregate statistics."""

    def __init__(self, base_config: SimulatorConfig, n_paths: int = 100):
        self.base_config = base_config
        self.n_paths = n_paths
        self.results: List[SimulationOutput] = []

    def run_all(self) -> None:
        self.results.clear()
        for i in range(self.n_paths):
            cfg = dataclasses.replace(self.base_config, seed=self.base_config.seed + i)
            sim = MarketSimulator(cfg)
            self.results.append(sim.run())

    def terminal_prices(self) -> FloatArray:
        """(n_paths, n_assets) of final prices."""
        return np.array([r.prices[-1] for r in self.results])

    def terminal_returns(self) -> FloatArray:
        arr = []
        for r in self.results:
            arr.append(r.prices[-1] / r.prices[0] - 1.0)
        return np.array(arr)

    def var(self, alpha: float = 0.05) -> FloatArray:
        """Value at Risk per asset."""
        tr = self.terminal_returns()
        return np.percentile(tr, alpha * 100, axis=0)

    def cvar(self, alpha: float = 0.05) -> FloatArray:
        """Conditional VaR (Expected Shortfall)."""
        tr = self.terminal_returns()
        var_val = self.var(alpha)
        mask = tr <= var_val
        if mask.any():
            return tr[mask].mean(axis=0)
        return var_val

    def probability_of_loss(self) -> FloatArray:
        tr = self.terminal_returns()
        return (tr < 0).mean(axis=0)

    def percentile_paths(self, percentiles: List[float] = None) -> Dict[str, FloatArray]:
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]
        all_prices = np.array([r.prices for r in self.results])  # (n_paths, n_steps, n_assets)
        result = {}
        for p in percentiles:
            result[f"p{p}"] = np.percentile(all_prices, p, axis=0)
        return result

    def max_drawdown_distribution(self) -> FloatArray:
        mdd = []
        for r in self.results:
            mdd.append(r.max_drawdown())
        return np.array(mdd)

    def sharpe_distribution(self) -> FloatArray:
        return np.array([r.sharpe_ratio() for r in self.results])


# ===================================================================
# 15. Correlation dynamics
# ===================================================================

class DynamicCorrelationTracker:
    """Track rolling and exponentially weighted correlation matrices."""

    def __init__(self, n_assets: int, lookback: int = 63, halflife: int = 21):
        self.n_assets = n_assets
        self.lookback = lookback
        self.halflife = halflife
        self._buffer: List[FloatArray] = []

    def update(self, returns: FloatArray) -> None:
        self._buffer.append(returns)
        if len(self._buffer) > self.lookback * 2:
            self._buffer = self._buffer[-self.lookback * 2 :]

    def rolling_correlation(self) -> FloatArray:
        if len(self._buffer) < self.lookback:
            return np.eye(self.n_assets)
        data = np.array(self._buffer[-self.lookback :])
        return np.corrcoef(data.T)

    def ewm_correlation(self) -> FloatArray:
        if len(self._buffer) < 5:
            return np.eye(self.n_assets)
        data = np.array(self._buffer)
        n = len(data)
        alpha = 1.0 - np.exp(-np.log(2) / self.halflife)
        weights = np.array([(1 - alpha) ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        mean = (weights[:, None] * data).sum(axis=0)
        centered = data - mean
        cov = (weights[:, None, None] * (centered[:, :, None] * centered[:, None, :])).sum(axis=0)
        std = np.sqrt(np.diag(cov) + 1e-12)
        corr = cov / (std[:, None] * std[None, :] + 1e-12)
        np.fill_diagonal(corr, 1.0)
        return corr

    def eigenvalue_spectrum(self) -> FloatArray:
        corr = self.rolling_correlation()
        eigvals = np.linalg.eigvalsh(corr)
        return np.sort(eigvals)[::-1]

    def absorption_ratio(self, n_components: int = 1) -> float:
        spectrum = self.eigenvalue_spectrum()
        return float(spectrum[:n_components].sum() / spectrum.sum())


# ===================================================================
# 16. Microstructure metrics
# ===================================================================

def compute_kyle_lambda(prices: FloatArray, volumes: FloatArray) -> float:
    dp = np.diff(prices)
    ofi = np.sign(dp) * np.sqrt(np.abs(volumes[1:]))
    valid = np.isfinite(ofi) & np.isfinite(dp) & (ofi != 0)
    if valid.sum() < 10:
        return 0.0
    X = ofi[valid].reshape(-1, 1)
    y = dp[valid]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(abs(beta[0]))


def compute_amihud_illiquidity(prices: FloatArray, volumes: FloatArray, window: int = 21) -> FloatArray:
    ret = np.abs(np.diff(np.log(prices + 1e-12)))
    dollar_vol = prices[1:] * volumes[1:]
    ratio = ret / (dollar_vol + 1e-12)
    n = len(ratio)
    result = np.full(n, np.nan)
    for i in range(window, n):
        result[i] = np.mean(ratio[i - window : i])
    return result


def compute_roll_spread(prices: FloatArray) -> float:
    dp = np.diff(prices)
    cov = np.cov(dp[:-1], dp[1:])[0, 1]
    if cov < 0:
        return 2.0 * np.sqrt(-cov)
    return 0.0


def compute_vpin(volumes: FloatArray, prices: FloatArray, n_buckets: int = 50) -> FloatArray:
    """Volume-synchronized PIN estimate."""
    dp = np.diff(prices)
    buy_vol = np.where(dp > 0, volumes[1:], 0.0)
    sell_vol = np.where(dp < 0, volumes[1:], 0.0)
    total_vol = volumes[1:]
    bucket_size = total_vol.sum() / n_buckets
    vpin = np.zeros(n_buckets)
    cum_vol = 0.0
    cum_buy = 0.0
    cum_sell = 0.0
    bucket_idx = 0
    for i in range(len(total_vol)):
        cum_vol += total_vol[i]
        cum_buy += buy_vol[i]
        cum_sell += sell_vol[i]
        if cum_vol >= bucket_size and bucket_idx < n_buckets:
            vpin[bucket_idx] = abs(cum_buy - cum_sell) / (cum_vol + 1e-12)
            cum_vol = 0.0
            cum_buy = 0.0
            cum_sell = 0.0
            bucket_idx += 1
    return vpin


# ===================================================================
# 17. Stress testing
# ===================================================================

@dataclass
class StressScenario:
    name: str
    price_shocks: FloatArray        # (n_assets,) multiplicative
    vol_shocks: FloatArray          # (n_assets,) multiplicative
    corr_override: FloatArray | None = None


class StressTester:
    """Apply stress scenarios to a portfolio."""

    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self.scenarios: List[StressScenario] = []

    def add_scenario(self, scenario: StressScenario) -> None:
        self.scenarios.append(scenario)

    def add_default_scenarios(self) -> None:
        n = self.n_assets
        self.scenarios.extend([
            StressScenario("equity_crash", np.full(n, 0.75), np.full(n, 2.5), build_correlation_matrix(n, 0.9)),
            StressScenario("rate_spike", np.full(n, 0.95), np.full(n, 1.5)),
            StressScenario("liquidity_crisis", np.full(n, 0.85), np.full(n, 3.0), build_correlation_matrix(n, 0.95)),
            StressScenario("sector_rotation", np.concatenate([np.full(n // 2, 0.90), np.full(n - n // 2, 1.10)]), np.full(n, 1.3)),
            StressScenario("flash_crash_recovery", np.full(n, 0.92), np.full(n, 4.0)),
        ])

    def apply(self, portfolio_weights: FloatArray, prices: FloatArray) -> Dict[str, float]:
        results = {}
        portfolio_value = np.dot(portfolio_weights, prices)
        for sc in self.scenarios:
            stressed_prices = prices * sc.price_shocks
            stressed_value = np.dot(portfolio_weights, stressed_prices)
            pnl_pct = (stressed_value - portfolio_value) / (portfolio_value + 1e-12)
            results[sc.name] = float(pnl_pct)
        return results

    def worst_case(self, portfolio_weights: FloatArray, prices: FloatArray) -> Tuple[str, float]:
        results = self.apply(portfolio_weights, prices)
        worst = min(results.items(), key=lambda x: x[1])
        return worst


# ===================================================================
# 18. Convenience runners
# ===================================================================

def quick_simulation(
    n_assets: int = 5,
    n_years: int = 10,
    seed: int = 42,
    regime_switching: bool = True,
) -> SimulationOutput:
    """One-liner to get a full simulation output."""
    cfg = SimulatorConfig(
        n_assets=n_assets,
        n_steps=252 * n_years,
        seed=seed,
        use_regime_switching=regime_switching,
        use_heston=True,
        use_jumps=True,
        use_flash_crash=True,
        use_events=True,
    )
    return MarketSimulator(cfg).run()


def run_with_order_book(
    n_steps: int = 1000,
    seed: int = 42,
) -> SimulationOutput:
    """Run simulation with order book and impact enabled."""
    cfg = SimulatorConfig(
        n_assets=1,
        n_steps=n_steps,
        seed=seed,
        use_order_book=True,
        use_impact=True,
        use_intraday=True,
        use_heston=True,
        use_jumps=False,
        use_regime_switching=False,
        use_flash_crash=True,
    )
    return MarketSimulator(cfg).run()


def compare_scenarios() -> Dict[str, SimulationOutput]:
    """Run all pre-built scenarios and return dict."""
    return {
        "bull": generate_bull_market(),
        "crisis": generate_crisis_scenario(),
        "low_vol": generate_low_vol_grind(),
        "choppy": generate_choppy_sideways(),
        "rate_hike": generate_rate_hike_scenario(),
    }


# ===================================================================
# 19. Analytics on simulation output
# ===================================================================

def compute_path_statistics(output: SimulationOutput) -> Dict[str, Any]:
    lr = output.log_returns
    sr = output.simple_returns
    stats_dict: Dict[str, Any] = {}
    stats_dict["annualized_return"] = float(lr.mean(axis=0).mean() * 252)
    stats_dict["annualized_vol"] = float(lr.std(axis=0).mean() * np.sqrt(252))
    stats_dict["sharpe"] = float(np.mean(output.sharpe_ratio()))
    stats_dict["max_drawdown"] = float(np.mean(output.max_drawdown()))
    stats_dict["skewness"] = float(np.mean(stats.skew(lr, axis=0)))
    stats_dict["kurtosis"] = float(np.mean(stats.kurtosis(lr, axis=0)))
    stats_dict["avg_spread"] = float(np.nanmean(output.spreads))
    stats_dict["n_flash_crashes"] = int(output.flash_crash_mask.sum())
    stats_dict["n_events"] = int(output.event_mask.sum())
    stats_dict["avg_correlation"] = float(np.mean(output.correlation_matrix()) - 1.0 / output.prices.shape[1])
    stats_dict["regime_counts"] = {int(k): int(v) for k, v in zip(*np.unique(output.regimes, return_counts=True))}
    return stats_dict


def compute_regime_statistics(output: SimulationOutput) -> Dict[int, Dict[str, float]]:
    lr = output.log_returns
    regimes = output.regimes[1:]
    result = {}
    for r in np.unique(regimes):
        mask = regimes == r
        if mask.sum() < 2:
            continue
        sub = lr[mask]
        result[int(r)] = {
            "mean_return": float(sub.mean() * 252),
            "volatility": float(sub.std() * np.sqrt(252)),
            "sharpe": float(sub.mean() / (sub.std() + 1e-12) * np.sqrt(252)),
            "skew": float(np.mean(stats.skew(sub, axis=0))),
            "count": int(mask.sum()),
        }
    return result


def compute_tail_statistics(output: SimulationOutput, threshold: float = 0.02) -> Dict[str, float]:
    lr = output.log_returns
    flat = lr.flatten()
    left_tail = flat[flat < -threshold]
    right_tail = flat[flat > threshold]
    return {
        "left_tail_count": len(left_tail),
        "right_tail_count": len(right_tail),
        "left_tail_mean": float(left_tail.mean()) if len(left_tail) > 0 else 0.0,
        "right_tail_mean": float(right_tail.mean()) if len(right_tail) > 0 else 0.0,
        "tail_ratio": len(right_tail) / max(len(left_tail), 1),
        "var_1pct": float(np.percentile(flat, 1)),
        "cvar_1pct": float(flat[flat <= np.percentile(flat, 1)].mean()) if len(flat) > 0 else 0.0,
        "var_5pct": float(np.percentile(flat, 5)),
        "cvar_5pct": float(flat[flat <= np.percentile(flat, 5)].mean()) if len(flat) > 0 else 0.0,
    }


# ===================================================================
# 20. Calibration helpers
# ===================================================================

def calibrate_heston_from_prices(
    prices: FloatArray, dt: float = 1.0 / 252.0
) -> HestonParams:
    """Simple method-of-moments Heston calibration from a single price series."""
    lr = np.diff(np.log(prices))
    rv_window = 21
    n = len(lr)
    rv = np.array([lr[max(0, i - rv_window) : i].var() * 252 for i in range(1, n + 1)])
    rv = np.maximum(rv, 1e-8)
    v0 = rv[-1]
    theta = rv.mean()
    dv = np.diff(rv)
    kappa_est = -np.polyfit(rv[:-1], dv, 1)[0] if len(rv) > 2 else 2.0
    kappa_est = np.clip(kappa_est, 0.1, 20.0)
    xi_est = np.std(dv) / (np.mean(np.sqrt(rv[:-1])) + 1e-8)
    xi_est = np.clip(xi_est, 0.01, 5.0)
    rho_est = np.corrcoef(lr[1:], dv)[0, 1] if len(dv) > 2 else -0.5
    return HestonParams(
        v0=float(v0),
        kappa=float(kappa_est),
        theta=float(theta),
        xi=float(xi_est),
        rho_sv=float(np.clip(rho_est, -0.99, 0.99)),
    )


def calibrate_merton_from_returns(returns: FloatArray) -> MertonJumpParams:
    """Simple calibration of jump params from return distribution."""
    threshold = 3.0 * returns.std()
    jumps = returns[np.abs(returns) > threshold]
    if len(jumps) < 3:
        return MertonJumpParams()
    lam_est = len(jumps) / len(returns) * 252
    mu_j = float(jumps.mean())
    sigma_j = float(jumps.std())
    return MertonJumpParams(lam=lam_est, mu_j=mu_j, sigma_j=sigma_j)


def calibrate_regime_from_returns(
    returns: FloatArray, n_regimes: int = 3, n_iter: int = 100
) -> Tuple[IntArray, List[Dict[str, float]]]:
    """Simple k-means style regime detection on rolling vol/return features."""
    window = 21
    n = len(returns)
    features = np.zeros((n - window, 2))
    for i in range(window, n):
        features[i - window, 0] = returns[i - window : i].mean() * 252
        features[i - window, 1] = returns[i - window : i].std() * np.sqrt(252)
    # Normalize
    mu_f = features.mean(axis=0)
    std_f = features.std(axis=0) + 1e-8
    features_norm = (features - mu_f) / std_f
    # K-means
    rng = np.random.default_rng(0)
    centroids = features_norm[rng.choice(len(features_norm), n_regimes, replace=False)]
    labels = np.zeros(len(features_norm), dtype=np.int64)
    for _ in range(n_iter):
        dists = np.array([np.linalg.norm(features_norm - c, axis=1) for c in centroids])
        labels = dists.argmin(axis=0)
        for k in range(n_regimes):
            mask = labels == k
            if mask.sum() > 0:
                centroids[k] = features_norm[mask].mean(axis=0)
    regime_stats = []
    for k in range(n_regimes):
        mask = labels == k
        if mask.sum() > 0:
            regime_stats.append({
                "mean_return": float(features[mask, 0].mean()),
                "mean_vol": float(features[mask, 1].mean()),
                "count": int(mask.sum()),
            })
        else:
            regime_stats.append({"mean_return": 0.0, "mean_vol": 0.0, "count": 0})
    full_labels = np.zeros(n, dtype=np.int64)
    full_labels[window:] = labels
    full_labels[:window] = labels[0]
    return full_labels, regime_stats


# ===================================================================
# __all__
# ===================================================================

__all__ = [
    "SimulatorConfig",
    "MarketSimulator",
    "SimulationOutput",
    "MonteCarloSimulator",
    "HestonParams",
    "MertonJumpParams",
    "RegimeSwitchConfig",
    "MarketRegime",
    "OrderBookSimulator",
    "OrderBookSnapshot",
    "MarketImpactModel",
    "ImpactParams",
    "FlashCrashEngine",
    "FlashCrashConfig",
    "EventEngine",
    "EventSchedule",
    "IntradayPatternEngine",
    "CentralBankEngine",
    "CentralBankConfig",
    "DynamicCorrelationTracker",
    "StressTester",
    "StressScenario",
    "quick_simulation",
    "run_with_order_book",
    "compare_scenarios",
    "compute_path_statistics",
    "compute_regime_statistics",
    "compute_tail_statistics",
    "calibrate_heston_from_prices",
    "calibrate_merton_from_returns",
    "calibrate_regime_from_returns",
    "compute_kyle_lambda",
    "compute_amihud_illiquidity",
    "compute_roll_spread",
    "compute_vpin",
]
