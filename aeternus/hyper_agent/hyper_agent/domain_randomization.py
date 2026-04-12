"""
domain_randomization.py — Domain Randomization Engine (DRE) for Hyper-Agent.

Injects synthetic microstructure noise and adversarial market participants into
training episodes to close the sim-to-real gap. Supports:

- Bid-ask spread noise (log-normal)
- Fill probability randomization (partial fills)
- Latency jitter injection (exponential distribution)
- Price impact randomization (vary Kyle lambda)
- Order book depth randomization
- Liquidity shock injection
- Regime randomization
- Auto-curriculum based on agent win-rate per scenario
- Randomization parameter annealing schedules
- Adversarial market participant injection

Integration hooks for MultiAssetTradingEnv.
"""

from __future__ import annotations

import math
import time
import random
import logging
import threading
import collections
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MarketRegime(Enum):
    CALM = auto()
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    VOLATILE = auto()
    CRASH = auto()
    RECOVERY = auto()
    ILLIQUID = auto()
    CRISIS = auto()


class AdversaryType(Enum):
    MOMENTUM_TRADER = auto()
    INFORMED_TRADER = auto()
    WASH_TRADER = auto()
    SPOOFING_TRADER = auto()
    ICEBERG_TRADER = auto()
    LATENCY_ARBITRAGEUR = auto()
    PREDATORY_HFT = auto()


class AnnealSchedule(Enum):
    LINEAR = auto()
    COSINE = auto()
    EXPONENTIAL = auto()
    STEP = auto()
    CYCLIC = auto()
    CONSTANT = auto()


class LiquidityShockType(Enum):
    SPREAD_WIDENING = auto()
    DEPTH_REMOVAL = auto()
    CROSS_ASSET_CRISIS = auto()
    FLASH_CRASH = auto()
    MARKET_FREEZE = auto()
    TICK_SIZE_CHANGE = auto()


# ---------------------------------------------------------------------------
# Config dataclass (50+ parameters)
# ---------------------------------------------------------------------------

@dataclass
class SpreadNoiseConfig:
    """Log-normal spread noise parameters."""
    enabled: bool = True
    base_multiplier_mean: float = 1.0       # mean of log-normal (in log space)
    base_multiplier_std: float = 0.3        # std of log-normal (in log space)
    min_multiplier: float = 0.5
    max_multiplier: float = 10.0
    autocorrelation: float = 0.7            # AR(1) coefficient for spread noise
    intraday_seasonality: bool = True       # U-shaped spread pattern
    seasonality_amplitude: float = 0.5
    tick_rounding: bool = True
    tick_size: float = 0.01


@dataclass
class FillRateConfig:
    """Partial fill randomization parameters."""
    enabled: bool = True
    base_fill_prob: float = 0.85            # probability order gets any fill
    partial_fill_prob: float = 0.4          # prob of partial (not full) fill
    fill_fraction_alpha: float = 2.0        # Beta distribution alpha
    fill_fraction_beta: float = 1.5         # Beta distribution beta
    queue_position_factor: float = 0.3      # how much queue position matters
    size_impact_factor: float = 0.2         # large orders get worse fills
    spread_fill_threshold: float = 0.001    # orders deeper in book fill faster
    market_impact_fill_delay: bool = True
    max_fill_delay_steps: int = 5


@dataclass
class LatencyConfig:
    """Latency jitter injection parameters."""
    enabled: bool = True
    base_latency_us: float = 100.0          # microseconds
    jitter_lambda: float = 0.01             # exponential distribution rate
    max_latency_us: float = 50_000.0        # 50ms cap
    processing_noise_std: float = 10.0
    network_congestion_prob: float = 0.05
    congestion_multiplier: float = 10.0
    packet_loss_prob: float = 0.001
    order_resequencing_prob: float = 0.01
    time_discretization_steps: int = 10


@dataclass
class PriceImpactConfig:
    """Kyle lambda (price impact) randomization."""
    enabled: bool = True
    kyle_lambda_min: float = 0.0001
    kyle_lambda_max: float = 0.005
    kyle_lambda_log_mean: float = -6.0      # log-normal mean
    kyle_lambda_log_std: float = 1.0
    temporary_impact_decay: float = 0.5     # mean reversion speed
    permanent_impact_fraction: float = 0.3  # fraction that is permanent
    cross_asset_impact_factor: float = 0.1
    volume_nonlinearity: float = 0.6        # power law exponent
    regime_dependence: bool = True          # lambda varies by regime


@dataclass
class OrderBookDepthConfig:
    """Order book depth randomization."""
    enabled: bool = True
    min_levels: int = 3
    max_levels: int = 20
    base_depth_mean: float = 100.0
    base_depth_std: float = 30.0
    depth_decay_rate: float = 0.7           # exponential decay across levels
    depth_imbalance_std: float = 0.3        # bid/ask depth imbalance
    refresh_rate_min: float = 0.1           # fraction of steps to refresh
    refresh_rate_max: float = 1.0
    correlated_depth: bool = True           # assets share depth shocks
    depth_correlation: float = 0.5


@dataclass
class LiquidityShockConfig:
    """Liquidity shock injection parameters."""
    enabled: bool = True
    shock_prob_per_step: float = 0.005
    spread_widening_min_factor: float = 5.0
    spread_widening_max_factor: float = 50.0
    depth_removal_fraction_min: float = 0.3
    depth_removal_fraction_max: float = 0.95
    shock_duration_steps_min: int = 5
    shock_duration_steps_max: int = 100
    recovery_half_life: float = 20.0
    cross_asset_contagion_prob: float = 0.4
    contagion_factor: float = 0.6
    flash_crash_magnitude: float = 0.05     # 5% price drop
    flash_crash_recovery_steps: int = 30
    market_freeze_prob: float = 0.001


@dataclass
class RegimeConfig:
    """Market regime randomization parameters."""
    enabled: bool = True
    initial_regime: Optional[MarketRegime] = None   # None = random
    regime_transition_prob: float = 0.01
    regime_min_duration: int = 50
    regime_max_duration: int = 500
    calm_vol: float = 0.01
    trending_drift: float = 0.002
    volatile_vol: float = 0.05
    crash_drift: float = -0.01
    crash_vol: float = 0.08
    recovery_drift: float = 0.003
    illiquid_spread_mult: float = 5.0
    crisis_correlation_boost: float = 0.4  # cross-asset correlation in crisis
    regime_weights: Dict[str, float] = field(default_factory=lambda: {
        "CALM": 0.40,
        "TRENDING_UP": 0.15,
        "TRENDING_DOWN": 0.15,
        "VOLATILE": 0.15,
        "CRASH": 0.05,
        "RECOVERY": 0.05,
        "ILLIQUID": 0.03,
        "CRISIS": 0.02,
    })


@dataclass
class AdversaryConfig:
    """Adversarial participant injection parameters."""
    enabled: bool = True
    max_adversaries: int = 5
    momentum_trader_prob: float = 0.3
    momentum_lookback: int = 10
    momentum_size_mean: float = 20.0
    informed_trader_prob: float = 0.2
    informed_signal_edge: float = 0.002     # price move they correctly predict
    informed_size_mean: float = 50.0
    wash_trader_prob: float = 0.1
    wash_trade_size: float = 100.0
    wash_trade_freq: int = 5
    spoofer_prob: float = 0.1
    spoof_order_size: float = 500.0
    spoof_cancel_delay: int = 3
    iceberg_prob: float = 0.15
    iceberg_visible_fraction: float = 0.1
    latency_arb_prob: float = 0.1
    predatory_hft_prob: float = 0.05
    predatory_hft_edge: float = 0.0001


@dataclass
class CurriculumConfig:
    """Auto-curriculum configuration."""
    enabled: bool = True
    eval_window: int = 100
    win_rate_threshold_increase: float = 0.65  # increase difficulty if above
    win_rate_threshold_decrease: float = 0.35  # decrease difficulty if below
    difficulty_step_size: float = 0.1
    min_difficulty: float = 0.0
    max_difficulty: float = 1.0
    initial_difficulty: float = 0.3
    per_scenario_tracking: bool = True
    scenario_types: List[str] = field(default_factory=lambda: [
        "spread_noise", "fill_randomization", "latency", "price_impact",
        "depth_random", "liquidity_shock", "regime_switch", "adversary",
    ])


@dataclass
class AnnealConfig:
    """Randomization annealing schedule configuration."""
    schedule_type: AnnealSchedule = AnnealSchedule.LINEAR
    total_steps: int = 1_000_000
    warmup_steps: int = 10_000
    initial_intensity: float = 0.5
    final_intensity: float = 1.0
    step_size: int = 50_000       # for STEP schedule
    step_decay: float = 0.9       # for STEP schedule
    cycle_length: int = 100_000   # for CYCLIC schedule
    min_intensity: float = 0.3
    max_intensity: float = 1.0


@dataclass
class DREConfig:
    """Master configuration for the Domain Randomization Engine."""
    # Sub-configs
    spread_noise: SpreadNoiseConfig = field(default_factory=SpreadNoiseConfig)
    fill_rate: FillRateConfig = field(default_factory=FillRateConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    price_impact: PriceImpactConfig = field(default_factory=PriceImpactConfig)
    order_book_depth: OrderBookDepthConfig = field(default_factory=OrderBookDepthConfig)
    liquidity_shock: LiquidityShockConfig = field(default_factory=LiquidityShockConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    adversary: AdversaryConfig = field(default_factory=AdversaryConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    anneal: AnnealConfig = field(default_factory=AnnealConfig)

    # Global switches
    enabled: bool = True
    seed: Optional[int] = None
    num_assets: int = 4
    num_agents: int = 8

    # Logging
    log_randomization: bool = False
    log_shocks: bool = True
    metrics_window: int = 1000

    # Integration
    env_step_callback: bool = True          # call DRE hooks on each env step
    reset_callback: bool = True             # call DRE hooks on env reset
    apply_to_observations: bool = True
    apply_to_rewards: bool = True
    apply_to_actions: bool = True


# ---------------------------------------------------------------------------
# Anneal scheduler
# ---------------------------------------------------------------------------

class AnnealScheduler:
    """Compute randomization intensity as a function of training step."""

    def __init__(self, config: AnnealConfig) -> None:
        self.config = config
        self._step = 0

    def step(self) -> None:
        self._step += 1

    def intensity(self) -> float:
        cfg = self.config
        t = max(0, self._step - cfg.warmup_steps)
        T = max(1, cfg.total_steps - cfg.warmup_steps)

        if cfg.schedule_type == AnnealSchedule.CONSTANT:
            val = cfg.initial_intensity
        elif cfg.schedule_type == AnnealSchedule.LINEAR:
            frac = min(1.0, t / T)
            val = cfg.initial_intensity + frac * (cfg.final_intensity - cfg.initial_intensity)
        elif cfg.schedule_type == AnnealSchedule.COSINE:
            frac = min(1.0, t / T)
            val = cfg.final_intensity + 0.5 * (cfg.initial_intensity - cfg.final_intensity) * (
                1.0 + math.cos(math.pi * frac)
            )
        elif cfg.schedule_type == AnnealSchedule.EXPONENTIAL:
            frac = min(1.0, t / T)
            val = cfg.initial_intensity * (cfg.final_intensity / cfg.initial_intensity) ** frac
        elif cfg.schedule_type == AnnealSchedule.STEP:
            num_steps = t // cfg.step_size
            val = cfg.initial_intensity * (cfg.step_decay ** num_steps)
        elif cfg.schedule_type == AnnealSchedule.CYCLIC:
            cycle_pos = (t % cfg.cycle_length) / cfg.cycle_length
            val = cfg.min_intensity + (cfg.max_intensity - cfg.min_intensity) * (
                0.5 - 0.5 * math.cos(2.0 * math.pi * cycle_pos)
            )
        else:
            val = cfg.initial_intensity

        if self._step < cfg.warmup_steps:
            warmup_frac = self._step / max(1, cfg.warmup_steps)
            val = cfg.initial_intensity * warmup_frac

        return float(np.clip(val, cfg.min_intensity, cfg.max_intensity))

    def reset_step(self) -> None:
        self._step = 0

    @property
    def current_step(self) -> int:
        return self._step


# ---------------------------------------------------------------------------
# Spread noise generator
# ---------------------------------------------------------------------------

class SpreadNoiseGenerator:
    """Generates log-normal bid-ask spread multipliers with AR(1) autocorrelation."""

    def __init__(self, config: SpreadNoiseConfig, rng: np.random.Generator) -> None:
        self.cfg = config
        self.rng = rng
        self._state: np.ndarray = np.zeros(1)   # AR(1) state (log-space)
        self._step = 0

    def reset(self, num_assets: int = 1) -> None:
        self._state = np.zeros(num_assets)
        self._step = 0

    def sample(self, num_assets: int = 1, time_of_day: float = 0.5) -> np.ndarray:
        """Return spread multipliers for each asset."""
        if not self.cfg.enabled:
            return np.ones(num_assets)

        cfg = self.cfg
        # AR(1) update in log space
        innovation = self.rng.normal(0.0, cfg.base_multiplier_std, size=num_assets)
        if len(self._state) != num_assets:
            self._state = np.zeros(num_assets)
        self._state = cfg.autocorrelation * self._state + (1 - cfg.autocorrelation) * innovation

        # Apply intraday seasonality (U-shape: wide at open/close, narrow midday)
        seasonal_factor = 0.0
        if cfg.intraday_seasonality:
            t = time_of_day  # 0=open, 1=close
            seasonal_factor = cfg.seasonality_amplitude * (
                4.0 * (t - 0.5) ** 2  # U-shape
            )

        log_multiplier = cfg.base_multiplier_mean + self._state + seasonal_factor
        multiplier = np.exp(log_multiplier)
        multiplier = np.clip(multiplier, cfg.min_multiplier, cfg.max_multiplier)
        self._step += 1
        return multiplier

    def apply_to_spread(
        self, spread: np.ndarray, time_of_day: float = 0.5
    ) -> np.ndarray:
        """Apply spread noise to an array of spreads."""
        mult = self.sample(len(spread), time_of_day)
        noisy = spread * mult
        if self.cfg.tick_rounding:
            noisy = np.round(noisy / self.cfg.tick_size) * self.cfg.tick_size
        return np.maximum(noisy, self.cfg.tick_size)


# ---------------------------------------------------------------------------
# Fill probability randomizer
# ---------------------------------------------------------------------------

class FillRateRandomizer:
    """Randomizes fill probability and fill fraction (partial fills)."""

    def __init__(self, config: FillRateConfig, rng: np.random.Generator) -> None:
        self.cfg = config
        self.rng = rng
        self._pending_fills: Dict[int, int] = {}  # order_id -> delay_steps_remaining

    def reset(self) -> None:
        self._pending_fills.clear()

    def sample_fill(
        self,
        order_id: int,
        order_size: float,
        queue_position: float = 0.5,
        spread: float = 0.01,
        mid_price: float = 100.0,
        limit_price: float = 100.0,
        intensity: float = 1.0,
    ) -> Tuple[bool, float, int]:
        """
        Sample whether an order gets filled, how much, and after how many steps.

        Returns:
            (filled: bool, fill_fraction: float, delay_steps: int)
        """
        if not self.cfg.enabled:
            return True, 1.0, 0

        cfg = self.cfg

        # Adjust fill prob by queue position (closer to top = better)
        queue_adj = 1.0 - cfg.queue_position_factor * queue_position

        # Adjust by order size (large orders fill worse)
        size_adj = max(0.5, 1.0 - cfg.size_impact_factor * math.log1p(order_size / 100.0))

        # Check if limit price is aggressive (inside spread)
        price_dist = abs(limit_price - mid_price) / max(spread, 1e-8)
        spread_adj = 1.0 if price_dist <= 0.5 else max(0.3, 1.0 - price_dist * 0.1)

        fill_prob = cfg.base_fill_prob * queue_adj * size_adj * spread_adj
        fill_prob = float(np.clip(fill_prob * (0.5 + 0.5 * intensity), 0.0, 1.0))

        # Determine if filled at all
        if self.rng.random() > fill_prob:
            return False, 0.0, 0

        # Determine fill fraction
        if self.rng.random() < cfg.partial_fill_prob:
            fill_frac = float(self.rng.beta(cfg.fill_fraction_alpha, cfg.fill_fraction_beta))
        else:
            fill_frac = 1.0

        # Delay
        delay = 0
        if cfg.market_impact_fill_delay and order_size > 100.0:
            delay = int(self.rng.integers(0, cfg.max_fill_delay_steps + 1))

        return True, fill_frac, delay

    def process_pending(self, current_step: int) -> List[int]:
        """Return order IDs whose delay has elapsed."""
        released = []
        to_delete = []
        for oid, remaining in self._pending_fills.items():
            if remaining <= 0:
                released.append(oid)
                to_delete.append(oid)
            else:
                self._pending_fills[oid] = remaining - 1
        for oid in to_delete:
            del self._pending_fills[oid]
        return released


# ---------------------------------------------------------------------------
# Latency jitter injector
# ---------------------------------------------------------------------------

class LatencyJitterInjector:
    """Injects exponentially-distributed latency jitter into order processing."""

    def __init__(self, config: LatencyConfig, rng: np.random.Generator) -> None:
        self.cfg = config
        self.rng = rng
        self._congestion_active: bool = False
        self._congestion_steps_remaining: int = 0

    def reset(self) -> None:
        self._congestion_active = False
        self._congestion_steps_remaining = 0

    def sample_latency_us(self, intensity: float = 1.0) -> float:
        """Sample a latency value in microseconds."""
        if not self.cfg.enabled:
            return self.cfg.base_latency_us

        cfg = self.cfg

        # Update congestion state
        if self._congestion_active:
            self._congestion_steps_remaining -= 1
            if self._congestion_steps_remaining <= 0:
                self._congestion_active = False
        elif self.rng.random() < cfg.network_congestion_prob * intensity:
            self._congestion_active = True
            self._congestion_steps_remaining = int(self.rng.integers(5, 50))

        # Exponential jitter
        jitter = self.rng.exponential(scale=1.0 / cfg.jitter_lambda)
        processing_noise = abs(self.rng.normal(0.0, cfg.processing_noise_std))

        total = cfg.base_latency_us + jitter + processing_noise

        if self._congestion_active:
            total *= cfg.congestion_multiplier

        total = float(np.clip(total, 0.0, cfg.max_latency_us))
        return total

    def should_drop_packet(self) -> bool:
        """Return True if the order/message should be dropped (packet loss)."""
        return bool(self.rng.random() < self.cfg.packet_loss_prob)

    def should_resequence(self) -> bool:
        """Return True if this order arrives out of sequence."""
        return bool(self.rng.random() < self.cfg.order_resequencing_prob)

    def latency_to_steps(self, latency_us: float, step_duration_us: float = 1000.0) -> int:
        """Convert latency in microseconds to discrete environment steps."""
        raw = latency_us / max(step_duration_us, 1.0)
        steps = int(math.ceil(raw / self.cfg.time_discretization_steps))
        return max(0, steps)


# ---------------------------------------------------------------------------
# Price impact randomizer
# ---------------------------------------------------------------------------

class PriceImpactRandomizer:
    """Randomizes Kyle lambda (price impact coefficient) across episodes/steps."""

    def __init__(self, config: PriceImpactConfig, rng: np.random.Generator) -> None:
        self.cfg = config
        self.rng = rng
        self._current_lambda: np.ndarray = np.array([])
        self._temporary_impact: np.ndarray = np.array([])

    def reset(self, num_assets: int, regime: MarketRegime = MarketRegime.CALM) -> None:
        self._current_lambda = self._sample_lambda(num_assets, regime)
        self._temporary_impact = np.zeros(num_assets)

    def _sample_lambda(self, num_assets: int, regime: MarketRegime) -> np.ndarray:
        cfg = self.cfg
        if not cfg.enabled:
            return np.full(num_assets, (cfg.kyle_lambda_min + cfg.kyle_lambda_max) / 2.0)

        lambdas = np.exp(
            self.rng.normal(cfg.kyle_lambda_log_mean, cfg.kyle_lambda_log_std, size=num_assets)
        )
        lambdas = np.clip(lambdas, cfg.kyle_lambda_min, cfg.kyle_lambda_max)

        if cfg.regime_dependence:
            mult = self._regime_lambda_multiplier(regime)
            lambdas *= mult

        return lambdas

    def _regime_lambda_multiplier(self, regime: MarketRegime) -> float:
        multipliers = {
            MarketRegime.CALM: 1.0,
            MarketRegime.TRENDING_UP: 1.2,
            MarketRegime.TRENDING_DOWN: 1.3,
            MarketRegime.VOLATILE: 1.8,
            MarketRegime.CRASH: 3.0,
            MarketRegime.RECOVERY: 1.5,
            MarketRegime.ILLIQUID: 4.0,
            MarketRegime.CRISIS: 5.0,
        }
        return multipliers.get(regime, 1.0)

    def compute_impact(
        self, trade_size: float, asset_id: int, sign: float = 1.0, intensity: float = 1.0
    ) -> Tuple[float, float]:
        """
        Compute (temporary_impact, permanent_impact) for a given trade.

        Uses square-root law: impact = lambda * sign * |size|^volume_nonlinearity
        """
        cfg = self.cfg
        if len(self._current_lambda) == 0 or asset_id >= len(self._current_lambda):
            lam = (cfg.kyle_lambda_min + cfg.kyle_lambda_max) / 2.0
        else:
            lam = self._current_lambda[asset_id] * intensity

        size_effect = abs(trade_size) ** cfg.volume_nonlinearity
        total_impact = lam * sign * size_effect
        perm = total_impact * cfg.permanent_impact_fraction
        temp = total_impact * (1.0 - cfg.permanent_impact_fraction)

        return float(temp), float(perm)

    def decay_temporary_impact(self, asset_id: int) -> float:
        """Decay temporary impact by mean reversion, return current value."""
        if asset_id < len(self._temporary_impact):
            self._temporary_impact[asset_id] *= (1.0 - self.cfg.temporary_impact_decay)
            return float(self._temporary_impact[asset_id])
        return 0.0

    def refresh(self, num_assets: int, regime: MarketRegime) -> None:
        """Re-sample lambdas (e.g., on regime change)."""
        self._current_lambda = self._sample_lambda(num_assets, regime)

    @property
    def current_lambda(self) -> np.ndarray:
        return self._current_lambda.copy()


# ---------------------------------------------------------------------------
# Order book depth randomizer
# ---------------------------------------------------------------------------

class OrderBookDepthRandomizer:
    """Randomizes order book depth levels and sizes."""

    def __init__(self, config: OrderBookDepthConfig, rng: np.random.Generator) -> None:
        self.cfg = config
        self.rng = rng
        self._bid_depths: np.ndarray = np.array([])
        self._ask_depths: np.ndarray = np.array([])
        self._num_levels: int = 10
        self._refresh_counter: int = 0
        self._refresh_rate: float = 0.5

    def reset(self, num_assets: int = 1) -> None:
        cfg = self.cfg
        self._num_levels = int(self.rng.integers(cfg.min_levels, cfg.max_levels + 1))
        self._refresh_rate = float(
            self.rng.uniform(cfg.refresh_rate_min, cfg.refresh_rate_max)
        )
        self._refresh_counter = 0
        self._sample_depths(num_assets)

    def _sample_depths(self, num_assets: int) -> None:
        cfg = self.cfg
        n = self._num_levels

        # Base depth per level (exponential decay)
        level_indices = np.arange(n)
        decay = cfg.depth_decay_rate ** level_indices
        base = self.rng.normal(cfg.base_depth_mean, cfg.base_depth_std, size=n)
        base = np.maximum(base, 1.0)

        # Per-asset depth (correlated if enabled)
        if cfg.correlated_depth and num_assets > 1:
            common_shock = self.rng.normal(0, 1)
            asset_shocks = (
                cfg.depth_correlation * common_shock
                + (1 - cfg.depth_correlation) * self.rng.normal(0, 1, size=num_assets)
            )
            asset_mult = np.exp(asset_shocks * 0.3)
        else:
            asset_mult = np.ones(num_assets)

        # Separate bid/ask with imbalance
        imbalance = self.rng.normal(0, cfg.depth_imbalance_std, size=num_assets)
        bid_mult = np.exp(imbalance)
        ask_mult = np.exp(-imbalance)

        # Shape: (num_assets, num_levels)
        self._bid_depths = np.outer(asset_mult * bid_mult, base * decay)
        self._ask_depths = np.outer(asset_mult * ask_mult, base * decay)
        self._bid_depths = np.maximum(self._bid_depths, 1.0)
        self._ask_depths = np.maximum(self._ask_depths, 1.0)

    def step(self, num_assets: int) -> None:
        self._refresh_counter += 1
        if self._refresh_counter >= 1.0 / max(self._refresh_rate, 1e-6):
            self._sample_depths(num_assets)
            self._refresh_counter = 0

    def get_depth(
        self, asset_id: int, side: str = "bid", level: int = 0
    ) -> float:
        """Get depth at a given level for an asset."""
        if not self.cfg.enabled:
            return self.cfg.base_depth_mean
        arr = self._bid_depths if side == "bid" else self._ask_depths
        if asset_id >= arr.shape[0] or level >= arr.shape[1]:
            return self.cfg.base_depth_mean
        return float(arr[asset_id, level])

    @property
    def num_levels(self) -> int:
        return self._num_levels


# ---------------------------------------------------------------------------
# Liquidity shock injector
# ---------------------------------------------------------------------------

@dataclass
class LiquidityShockState:
    active: bool = False
    shock_type: LiquidityShockType = LiquidityShockType.SPREAD_WIDENING
    affected_assets: List[int] = field(default_factory=list)
    spread_multiplier: float = 1.0
    depth_fraction_remaining: float = 1.0
    steps_remaining: int = 0
    total_duration: int = 0

    def progress(self) -> float:
        if self.total_duration == 0:
            return 1.0
        return 1.0 - self.steps_remaining / self.total_duration


class LiquidityShockInjector:
    """Injects sudden liquidity shocks into the trading environment."""

    def __init__(
        self,
        config: LiquidityShockConfig,
        rng: np.random.Generator,
        num_assets: int = 4,
    ) -> None:
        self.cfg = config
        self.rng = rng
        self.num_assets = num_assets
        self._shocks: List[LiquidityShockState] = []
        self._shock_history: List[Dict[str, Any]] = []
        self._step = 0

    def reset(self) -> None:
        self._shocks.clear()
        self._step = 0

    def step(self, intensity: float = 1.0) -> List[LiquidityShockState]:
        """Advance one step: possibly inject new shock, decay existing ones."""
        self._step += 1
        cfg = self.cfg

        # Possibly inject new shock
        shock_prob = cfg.shock_prob_per_step * intensity
        if self.rng.random() < shock_prob and cfg.enabled:
            new_shock = self._sample_shock()
            self._shocks.append(new_shock)
            self._shock_history.append({
                "step": self._step,
                "type": new_shock.shock_type.name,
                "assets": new_shock.affected_assets.copy(),
                "multiplier": new_shock.spread_multiplier,
            })
            if cfg.shock_prob_per_step > 0 or cfg.market_freeze_prob > 0:
                logger.debug("Liquidity shock: %s at step %d", new_shock.shock_type.name, self._step)

        # Decay existing shocks
        active_shocks = []
        for shock in self._shocks:
            shock.steps_remaining -= 1
            if shock.steps_remaining > 0:
                # Recovery: linearly interpolate back
                prog = shock.progress()
                decay = math.exp(-prog * 3.0 / max(cfg.recovery_half_life, 1.0))
                shock.spread_multiplier = max(1.0, shock.spread_multiplier * (1.0 + decay * 0.01))
                shock.depth_fraction_remaining = min(
                    1.0, shock.depth_fraction_remaining + (1.0 - shock.depth_fraction_remaining) * 0.05
                )
                active_shocks.append(shock)

        self._shocks = active_shocks
        return self._shocks

    def _sample_shock(self) -> LiquidityShockState:
        cfg = self.cfg

        # Pick shock type
        if self.rng.random() < cfg.market_freeze_prob:
            shock_type = LiquidityShockType.MARKET_FREEZE
        else:
            shock_type = self.rng.choice([  # type: ignore[arg-type]
                LiquidityShockType.SPREAD_WIDENING,
                LiquidityShockType.DEPTH_REMOVAL,
                LiquidityShockType.CROSS_ASSET_CRISIS,
                LiquidityShockType.FLASH_CRASH,
            ], p=[0.4, 0.3, 0.2, 0.1])

        # Pick affected assets
        if shock_type == LiquidityShockType.CROSS_ASSET_CRISIS:
            n_affected = self.rng.integers(2, self.num_assets + 1)
            affected = list(self.rng.choice(self.num_assets, size=n_affected, replace=False))
        else:
            affected = [int(self.rng.integers(0, self.num_assets))]
            if shock_type != LiquidityShockType.MARKET_FREEZE:
                if self.rng.random() < cfg.cross_asset_contagion_prob:
                    extra = self.rng.choice(self.num_assets)
                    if extra not in affected:
                        affected.append(int(extra))

        # Shock parameters
        spread_mult = float(
            self.rng.uniform(cfg.spread_widening_min_factor, cfg.spread_widening_max_factor)
        )
        depth_remaining = float(
            1.0 - self.rng.uniform(cfg.depth_removal_fraction_min, cfg.depth_removal_fraction_max)
        )
        duration = int(
            self.rng.integers(cfg.shock_duration_steps_min, cfg.shock_duration_steps_max + 1)
        )

        if shock_type == LiquidityShockType.MARKET_FREEZE:
            spread_mult = 50.0
            depth_remaining = 0.05

        return LiquidityShockState(
            active=True,
            shock_type=shock_type,
            affected_assets=affected,
            spread_multiplier=spread_mult,
            depth_fraction_remaining=depth_remaining,
            steps_remaining=duration,
            total_duration=duration,
        )

    def get_spread_multiplier(self, asset_id: int) -> float:
        """Return aggregate spread multiplier from all active shocks for an asset."""
        mult = 1.0
        for shock in self._shocks:
            if asset_id in shock.affected_assets:
                mult = max(mult, shock.spread_multiplier)
        return mult

    def get_depth_fraction(self, asset_id: int) -> float:
        """Return the fraction of depth remaining for an asset."""
        frac = 1.0
        for shock in self._shocks:
            if asset_id in shock.affected_assets:
                frac = min(frac, shock.depth_fraction_remaining)
        return frac

    @property
    def active_shocks(self) -> List[LiquidityShockState]:
        return [s for s in self._shocks if s.steps_remaining > 0]

    @property
    def shock_history(self) -> List[Dict[str, Any]]:
        return self._shock_history.copy()


# ---------------------------------------------------------------------------
# Regime manager
# ---------------------------------------------------------------------------

class RegimeManager:
    """Manages market regime transitions during training episodes."""

    def __init__(
        self,
        config: RegimeConfig,
        rng: np.random.Generator,
    ) -> None:
        self.cfg = config
        self.rng = rng
        self._regime: MarketRegime = MarketRegime.CALM
        self._steps_in_regime: int = 0
        self._regime_duration: int = 0
        self._regime_history: List[Tuple[int, str]] = []
        self._step: int = 0

    def reset(self) -> MarketRegime:
        cfg = self.cfg
        if cfg.initial_regime is not None:
            self._regime = cfg.initial_regime
        else:
            self._regime = self._sample_regime()
        self._steps_in_regime = 0
        self._regime_duration = self._sample_duration()
        self._regime_history.clear()
        self._step = 0
        self._regime_history.append((0, self._regime.name))
        return self._regime

    def _sample_regime(self) -> MarketRegime:
        cfg = self.cfg
        names = list(cfg.regime_weights.keys())
        weights = [cfg.regime_weights[n] for n in names]
        total = sum(weights)
        weights = [w / total for w in weights]
        chosen = self.rng.choice(len(names), p=weights)  # type: ignore[arg-type]
        return MarketRegime[names[chosen]]

    def _sample_duration(self) -> int:
        return int(
            self.rng.integers(self.cfg.regime_min_duration, self.cfg.regime_max_duration + 1)
        )

    def step(self, intensity: float = 1.0) -> Tuple[MarketRegime, bool]:
        """
        Advance one step. Returns (current_regime, regime_changed).
        Regime switches happen either:
          (a) when duration expires, or
          (b) randomly with transition_prob.
        """
        self._step += 1
        self._steps_in_regime += 1
        changed = False

        duration_expired = self._steps_in_regime >= self._regime_duration
        random_switch = (
            self.rng.random() < self.cfg.regime_transition_prob * intensity
            and self._steps_in_regime >= self.cfg.regime_min_duration
        )

        if duration_expired or random_switch:
            old_regime = self._regime
            self._regime = self._sample_regime()
            while self._regime == old_regime:
                self._regime = self._sample_regime()
            self._steps_in_regime = 0
            self._regime_duration = self._sample_duration()
            self._regime_history.append((self._step, self._regime.name))
            changed = True

        return self._regime, changed

    def get_volatility(self) -> float:
        """Return volatility parameter for current regime."""
        cfg = self.cfg
        vols = {
            MarketRegime.CALM: cfg.calm_vol,
            MarketRegime.TRENDING_UP: cfg.calm_vol * 1.5,
            MarketRegime.TRENDING_DOWN: cfg.calm_vol * 1.7,
            MarketRegime.VOLATILE: cfg.volatile_vol,
            MarketRegime.CRASH: cfg.crash_vol,
            MarketRegime.RECOVERY: cfg.calm_vol * 2.0,
            MarketRegime.ILLIQUID: cfg.calm_vol * 0.8,
            MarketRegime.CRISIS: cfg.crash_vol * 1.5,
        }
        return vols.get(self._regime, cfg.calm_vol)

    def get_drift(self) -> float:
        """Return price drift for current regime."""
        cfg = self.cfg
        drifts = {
            MarketRegime.CALM: 0.0,
            MarketRegime.TRENDING_UP: cfg.trending_drift,
            MarketRegime.TRENDING_DOWN: -cfg.trending_drift,
            MarketRegime.VOLATILE: 0.0,
            MarketRegime.CRASH: cfg.crash_drift,
            MarketRegime.RECOVERY: cfg.recovery_drift,
            MarketRegime.ILLIQUID: 0.0,
            MarketRegime.CRISIS: cfg.crash_drift * 1.5,
        }
        return drifts.get(self._regime, 0.0)

    def get_correlation_boost(self) -> float:
        """Additional cross-asset correlation in this regime."""
        if self._regime in (MarketRegime.CRASH, MarketRegime.CRISIS):
            return self.cfg.crisis_correlation_boost
        return 0.0

    @property
    def current_regime(self) -> MarketRegime:
        return self._regime

    @property
    def regime_history(self) -> List[Tuple[int, str]]:
        return self._regime_history.copy()

    @property
    def steps_in_current_regime(self) -> int:
        return self._steps_in_regime


# ---------------------------------------------------------------------------
# Adversarial market participant
# ---------------------------------------------------------------------------

@dataclass
class AdversaryState:
    adversary_type: AdversaryType
    asset_id: int
    active: bool = True
    position: float = 0.0
    cash: float = 0.0
    last_action_step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdversarialParticipantInjector:
    """Injects adversarial market participants into training environment."""

    def __init__(
        self,
        config: AdversaryConfig,
        rng: np.random.Generator,
        num_assets: int = 4,
    ) -> None:
        self.cfg = config
        self.rng = rng
        self.num_assets = num_assets
        self._adversaries: List[AdversaryState] = []
        self._step = 0

    def reset(self) -> None:
        self._adversaries.clear()
        self._step = 0
        if not self.cfg.enabled:
            return
        # Inject initial adversaries
        n = int(self.rng.integers(0, self.cfg.max_adversaries + 1))
        for _ in range(n):
            adv = self._spawn_adversary()
            if adv is not None:
                self._adversaries.append(adv)

    def _spawn_adversary(self) -> Optional[AdversaryState]:
        cfg = self.cfg
        probs = [
            cfg.momentum_trader_prob,
            cfg.informed_trader_prob,
            cfg.wash_trader_prob,
            cfg.spoofer_prob,
            cfg.iceberg_prob,
            cfg.latency_arb_prob,
            cfg.predatory_hft_prob,
        ]
        types = [
            AdversaryType.MOMENTUM_TRADER,
            AdversaryType.INFORMED_TRADER,
            AdversaryType.WASH_TRADER,
            AdversaryType.SPOOFING_TRADER,
            AdversaryType.ICEBERG_TRADER,
            AdversaryType.LATENCY_ARBITRAGEUR,
            AdversaryType.PREDATORY_HFT,
        ]
        total = sum(probs)
        if total <= 0:
            return None
        norm_probs = [p / total for p in probs]
        idx = int(self.rng.choice(len(types), p=norm_probs))  # type: ignore[arg-type]
        asset_id = int(self.rng.integers(0, self.num_assets))
        meta: Dict[str, Any] = {}
        atype = types[idx]
        if atype == AdversaryType.MOMENTUM_TRADER:
            meta["lookback"] = int(self.rng.integers(5, cfg.momentum_lookback * 2))
            meta["size"] = float(self.rng.exponential(cfg.momentum_size_mean))
            meta["price_history"] = []
        elif atype == AdversaryType.INFORMED_TRADER:
            meta["signal_edge"] = cfg.informed_signal_edge * float(self.rng.uniform(0.5, 2.0))
            meta["size"] = float(self.rng.exponential(cfg.informed_size_mean))
        elif atype == AdversaryType.WASH_TRADER:
            meta["size"] = cfg.wash_trade_size
            meta["freq"] = cfg.wash_trade_freq
        elif atype == AdversaryType.SPOOFING_TRADER:
            meta["spoof_size"] = cfg.spoof_order_size
            meta["cancel_delay"] = cfg.spoof_cancel_delay
            meta["pending_spoofs"] = []
        return AdversaryState(
            adversary_type=atype,
            asset_id=asset_id,
            metadata=meta,
        )

    def step(
        self,
        mid_prices: np.ndarray,
        spreads: np.ndarray,
        intensity: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Advance adversary step. Returns list of orders injected.

        Each order dict: {type, asset_id, side, price, size, adversary_type}
        """
        self._step += 1
        injected_orders: List[Dict[str, Any]] = []

        if not self.cfg.enabled:
            return injected_orders

        for adv in self._adversaries:
            if not adv.active:
                continue
            orders = self._adversary_action(adv, mid_prices, spreads, intensity)
            injected_orders.extend(orders)
            adv.last_action_step = self._step

        # Randomly spawn new adversaries
        if (
            len(self._adversaries) < self.cfg.max_adversaries
            and self.rng.random() < 0.01 * intensity
        ):
            new_adv = self._spawn_adversary()
            if new_adv is not None:
                self._adversaries.append(new_adv)

        return injected_orders

    def _adversary_action(
        self,
        adv: AdversaryState,
        mid_prices: np.ndarray,
        spreads: np.ndarray,
        intensity: float,
    ) -> List[Dict[str, Any]]:
        orders: List[Dict[str, Any]] = []
        aid = adv.asset_id
        if aid >= len(mid_prices):
            return orders
        mid = mid_prices[aid]
        spread = spreads[aid]

        atype = adv.adversary_type
        cfg = self.cfg

        if atype == AdversaryType.MOMENTUM_TRADER:
            hist = adv.metadata.get("price_history", [])
            hist.append(mid)
            lookback = adv.metadata.get("lookback", 10)
            if len(hist) > lookback:
                hist.pop(0)
            adv.metadata["price_history"] = hist
            if len(hist) >= lookback:
                momentum = (hist[-1] - hist[0]) / (hist[0] + 1e-8)
                if abs(momentum) > 0.001 * intensity:
                    side = "buy" if momentum > 0 else "sell"
                    price = mid + (spread / 2.0) * (1 if side == "buy" else -1)
                    orders.append({
                        "type": "limit",
                        "asset_id": aid,
                        "side": side,
                        "price": price,
                        "size": adv.metadata.get("size", 20.0),
                        "adversary_type": atype.name,
                    })

        elif atype == AdversaryType.INFORMED_TRADER:
            edge = adv.metadata.get("signal_edge", cfg.informed_signal_edge)
            if self.rng.random() < 0.3 * intensity:
                direction = 1 if self.rng.random() < 0.5 + edge * 100 else -1
                side = "buy" if direction > 0 else "sell"
                price = mid + direction * spread * 0.5
                orders.append({
                    "type": "limit",
                    "asset_id": aid,
                    "side": side,
                    "price": price,
                    "size": adv.metadata.get("size", 50.0),
                    "adversary_type": atype.name,
                })

        elif atype == AdversaryType.WASH_TRADER:
            freq = adv.metadata.get("freq", cfg.wash_trade_freq)
            if self._step % max(freq, 1) == 0:
                size = adv.metadata.get("size", cfg.wash_trade_size)
                # Buy and sell simultaneously
                orders.append({
                    "type": "limit",
                    "asset_id": aid,
                    "side": "buy",
                    "price": mid + spread * 0.5,
                    "size": size,
                    "adversary_type": atype.name,
                })
                orders.append({
                    "type": "limit",
                    "asset_id": aid,
                    "side": "sell",
                    "price": mid - spread * 0.5,
                    "size": size,
                    "adversary_type": atype.name,
                })

        elif atype == AdversaryType.SPOOFING_TRADER:
            spoof_size = adv.metadata.get("spoof_size", cfg.spoof_order_size)
            cancel_delay = adv.metadata.get("cancel_delay", cfg.spoof_cancel_delay)
            pending = adv.metadata.get("pending_spoofs", [])
            # Cancel old spoofs
            new_pending = [p for p in pending if self._step - p["step"] < cancel_delay]
            if len(new_pending) < 2 and self.rng.random() < 0.2 * intensity:
                # Place large spoof order far from mid
                side = "buy" if self.rng.random() < 0.5 else "sell"
                offset = 5.0 * spread if side == "buy" else -5.0 * spread
                price = mid + offset
                orders.append({
                    "type": "limit",
                    "asset_id": aid,
                    "side": side,
                    "price": price,
                    "size": spoof_size,
                    "adversary_type": atype.name,
                    "is_spoof": True,
                })
                new_pending.append({"step": self._step, "side": side})
            adv.metadata["pending_spoofs"] = new_pending

        elif atype == AdversaryType.ICEBERG_TRADER:
            visible_frac = cfg.iceberg_visible_fraction
            full_size = adv.metadata.get("size", 200.0)
            visible_size = full_size * visible_frac
            if self.rng.random() < 0.1 * intensity:
                side = "buy" if self.rng.random() < 0.5 else "sell"
                price = mid + (spread * 0.1 if side == "buy" else -spread * 0.1)
                orders.append({
                    "type": "limit",
                    "asset_id": aid,
                    "side": side,
                    "price": price,
                    "size": visible_size,
                    "adversary_type": atype.name,
                    "is_iceberg": True,
                    "hidden_size": full_size - visible_size,
                })

        elif atype == AdversaryType.PREDATORY_HFT:
            edge = cfg.predatory_hft_edge
            if self.rng.random() < 0.5 * intensity:
                side = "buy" if self.rng.random() < 0.5 + edge else "sell"
                price = mid
                orders.append({
                    "type": "market",
                    "asset_id": aid,
                    "side": side,
                    "price": price,
                    "size": 1.0,
                    "adversary_type": atype.name,
                })

        return orders

    @property
    def num_active_adversaries(self) -> int:
        return sum(1 for a in self._adversaries if a.active)

    @property
    def adversary_types(self) -> List[str]:
        return [a.adversary_type.name for a in self._adversaries if a.active]


# ---------------------------------------------------------------------------
# Auto-curriculum tracker
# ---------------------------------------------------------------------------

class AutoCurriculum:
    """
    Tracks agent win rate per scenario type and adjusts difficulty.

    High win rate → increase difficulty (more randomization).
    Low win rate → decrease difficulty (less randomization).
    """

    def __init__(self, config: CurriculumConfig) -> None:
        self.cfg = config
        self._difficulties: Dict[str, float] = {
            s: config.initial_difficulty for s in config.scenario_types
        }
        self._outcomes: Dict[str, collections.deque] = {
            s: collections.deque(maxlen=config.eval_window)
            for s in config.scenario_types
        }
        self._global_difficulty: float = config.initial_difficulty
        self._update_count: int = 0

    def record_outcome(self, scenario_type: str, win: bool) -> None:
        """Record a win/loss for a scenario type."""
        if not self.cfg.enabled:
            return
        if scenario_type not in self._outcomes:
            self._outcomes[scenario_type] = collections.deque(maxlen=self.cfg.eval_window)
            self._difficulties[scenario_type] = self.cfg.initial_difficulty
        self._outcomes[scenario_type].append(1.0 if win else 0.0)
        self._update_difficulty(scenario_type)
        self._update_count += 1

    def _update_difficulty(self, scenario_type: str) -> None:
        outcomes = self._outcomes[scenario_type]
        if len(outcomes) < 10:
            return
        win_rate = float(np.mean(list(outcomes)))
        current = self._difficulties[scenario_type]

        if win_rate > self.cfg.win_rate_threshold_increase:
            new_diff = min(
                self.cfg.max_difficulty,
                current + self.cfg.difficulty_step_size,
            )
        elif win_rate < self.cfg.win_rate_threshold_decrease:
            new_diff = max(
                self.cfg.min_difficulty,
                current - self.cfg.difficulty_step_size,
            )
        else:
            new_diff = current

        self._difficulties[scenario_type] = new_diff

        # Update global difficulty as weighted average
        self._global_difficulty = float(np.mean(list(self._difficulties.values())))

    def get_difficulty(self, scenario_type: Optional[str] = None) -> float:
        """Get current difficulty for a scenario or global difficulty."""
        if not self.cfg.enabled:
            return self.cfg.initial_difficulty
        if scenario_type is not None and scenario_type in self._difficulties:
            return self._difficulties[scenario_type]
        return self._global_difficulty

    def get_win_rate(self, scenario_type: str) -> float:
        outcomes = self._outcomes.get(scenario_type, collections.deque())
        if not outcomes:
            return 0.5
        return float(np.mean(list(outcomes)))

    def get_all_difficulties(self) -> Dict[str, float]:
        return self._difficulties.copy()

    def get_curriculum_summary(self) -> Dict[str, Any]:
        return {
            "global_difficulty": self._global_difficulty,
            "per_scenario": {
                s: {
                    "difficulty": self._difficulties.get(s, self.cfg.initial_difficulty),
                    "win_rate": self.get_win_rate(s),
                    "num_episodes": len(self._outcomes.get(s, [])),
                }
                for s in self.cfg.scenario_types
            },
            "total_updates": self._update_count,
        }

    def reset_statistics(self) -> None:
        for s in self.cfg.scenario_types:
            self._outcomes[s].clear()
        self._global_difficulty = self.cfg.initial_difficulty


# ---------------------------------------------------------------------------
# DRE observation augmentation
# ---------------------------------------------------------------------------

class ObservationAugmentor:
    """Applies randomization to agent observations."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def add_observation_noise(
        self,
        obs: np.ndarray,
        noise_std: float = 0.01,
        dropout_prob: float = 0.05,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """Add Gaussian noise and random dropout to observations."""
        noisy = obs + self.rng.normal(0.0, noise_std * intensity, size=obs.shape)
        mask = self.rng.random(size=obs.shape) < dropout_prob * intensity
        noisy = np.where(mask, 0.0, noisy)
        return noisy

    def add_stale_observation(
        self,
        obs: np.ndarray,
        obs_history: List[np.ndarray],
        stale_prob: float = 0.02,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """Occasionally return a stale (old) observation."""
        if obs_history and self.rng.random() < stale_prob * intensity:
            lag = int(self.rng.integers(1, min(len(obs_history) + 1, 10)))
            return obs_history[-lag].copy()
        return obs

    def apply_regime_feature_noise(
        self,
        obs: np.ndarray,
        regime: MarketRegime,
        intensity: float = 1.0,
    ) -> np.ndarray:
        """Add regime-specific feature noise."""
        regime_noise_scales = {
            MarketRegime.CALM: 0.001,
            MarketRegime.TRENDING_UP: 0.005,
            MarketRegime.TRENDING_DOWN: 0.005,
            MarketRegime.VOLATILE: 0.02,
            MarketRegime.CRASH: 0.05,
            MarketRegime.RECOVERY: 0.015,
            MarketRegime.ILLIQUID: 0.01,
            MarketRegime.CRISIS: 0.08,
        }
        scale = regime_noise_scales.get(regime, 0.01) * intensity
        return obs + self.rng.normal(0.0, scale, size=obs.shape)


# ---------------------------------------------------------------------------
# DRE metrics tracker
# ---------------------------------------------------------------------------

class DREMetricsTracker:
    """Tracks statistics about randomization applied during training."""

    def __init__(self, window: int = 1000) -> None:
        self._window = window
        self._spread_mults: collections.deque = collections.deque(maxlen=window)
        self._fill_rates: collections.deque = collections.deque(maxlen=window)
        self._latencies_us: collections.deque = collections.deque(maxlen=window)
        self._kyle_lambdas: collections.deque = collections.deque(maxlen=window)
        self._shocks: int = 0
        self._regime_changes: int = 0
        self._adversary_orders: int = 0
        self._curriculum_adjustments: int = 0
        self._step: int = 0

    def record_spread_mult(self, mult: float) -> None:
        self._spread_mults.append(mult)

    def record_fill_rate(self, rate: float) -> None:
        self._fill_rates.append(rate)

    def record_latency(self, lat_us: float) -> None:
        self._latencies_us.append(lat_us)

    def record_kyle_lambda(self, lam: float) -> None:
        self._kyle_lambdas.append(lam)

    def record_shock(self) -> None:
        self._shocks += 1

    def record_regime_change(self) -> None:
        self._regime_changes += 1

    def record_adversary_orders(self, n: int) -> None:
        self._adversary_orders += n

    def record_curriculum_adjustment(self) -> None:
        self._curriculum_adjustments += 1

    def step(self) -> None:
        self._step += 1

    def get_summary(self) -> Dict[str, Any]:
        def safe_mean(d: collections.deque) -> float:
            return float(np.mean(list(d))) if d else 0.0
        def safe_std(d: collections.deque) -> float:
            return float(np.std(list(d))) if len(d) > 1 else 0.0

        return {
            "step": self._step,
            "spread_mult_mean": safe_mean(self._spread_mults),
            "spread_mult_std": safe_std(self._spread_mults),
            "fill_rate_mean": safe_mean(self._fill_rates),
            "latency_us_mean": safe_mean(self._latencies_us),
            "latency_us_p99": float(np.percentile(list(self._latencies_us), 99)) if self._latencies_us else 0.0,
            "kyle_lambda_mean": safe_mean(self._kyle_lambdas),
            "total_shocks": self._shocks,
            "total_regime_changes": self._regime_changes,
            "total_adversary_orders": self._adversary_orders,
            "total_curriculum_adjustments": self._curriculum_adjustments,
        }


# ---------------------------------------------------------------------------
# Domain Randomization Engine (main class)
# ---------------------------------------------------------------------------

class DomainRandomizationEngine:
    """
    Master Domain Randomization Engine for Hyper-Agent sim-to-real gap closure.

    Orchestrates all sub-modules:
    - SpreadNoiseGenerator
    - FillRateRandomizer
    - LatencyJitterInjector
    - PriceImpactRandomizer
    - OrderBookDepthRandomizer
    - LiquidityShockInjector
    - RegimeManager
    - AdversarialParticipantInjector
    - AutoCurriculum
    - AnnealScheduler
    - ObservationAugmentor
    - DREMetricsTracker
    """

    def __init__(self, config: Optional[DREConfig] = None) -> None:
        self.cfg = config or DREConfig()
        seed = self.cfg.seed
        self.rng = np.random.default_rng(seed)

        # Sub-modules
        self.spread_noise = SpreadNoiseGenerator(self.cfg.spread_noise, self.rng)
        self.fill_rate = FillRateRandomizer(self.cfg.fill_rate, self.rng)
        self.latency = LatencyJitterInjector(self.cfg.latency, self.rng)
        self.price_impact = PriceImpactRandomizer(self.cfg.price_impact, self.rng)
        self.depth = OrderBookDepthRandomizer(self.cfg.order_book_depth, self.rng)
        self.shock = LiquidityShockInjector(
            self.cfg.liquidity_shock, self.rng, self.cfg.num_assets
        )
        self.regime = RegimeManager(self.cfg.regime, self.rng)
        self.adversary = AdversarialParticipantInjector(
            self.cfg.adversary, self.rng, self.cfg.num_assets
        )
        self.curriculum = AutoCurriculum(self.cfg.curriculum)
        self.annealer = AnnealScheduler(self.cfg.anneal)
        self.augmentor = ObservationAugmentor(self.rng)
        self.metrics = DREMetricsTracker(self.cfg.metrics_window)

        self._current_regime: MarketRegime = MarketRegime.CALM
        self._step: int = 0
        self._obs_history: List[np.ndarray] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        """Reset all sub-modules for a new episode. Returns initial randomization state."""
        with self._lock:
            n = self.cfg.num_assets
            self.spread_noise.reset(n)
            self.fill_rate.reset()
            self.latency.reset()
            self.price_impact.reset(n)
            self.depth.reset(n)
            self.shock.reset()
            self._current_regime = self.regime.reset()
            self.adversary.reset()
            self._obs_history.clear()
            self._step = 0

        return self._get_state()

    def _get_state(self) -> Dict[str, Any]:
        return {
            "regime": self._current_regime.name,
            "regime_volatility": self.regime.get_volatility(),
            "regime_drift": self.regime.get_drift(),
            "annealer_intensity": self.annealer.intensity(),
            "global_difficulty": self.curriculum.get_difficulty(),
            "num_active_adversaries": self.adversary.num_active_adversaries,
            "num_active_shocks": len(self.shock.active_shocks),
        }

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        mid_prices: np.ndarray,
        spreads: np.ndarray,
        time_of_day: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Advance one environment step. Returns randomization outputs for the env.

        Args:
            mid_prices: current mid prices per asset
            spreads: current bid-ask spreads per asset
            time_of_day: fractional day time [0, 1]

        Returns:
            dict with randomized parameters for this step
        """
        with self._lock:
            self._step += 1
            intensity = self.annealer.intensity()
            self.annealer.step()
            self.metrics.step()

            n = self.cfg.num_assets

            # Regime update
            self._current_regime, regime_changed = self.regime.step(intensity)
            if regime_changed:
                self.price_impact.refresh(n, self._current_regime)
                self.metrics.record_regime_change()

            # Spread noise
            spread_mults = self.spread_noise.sample(n, time_of_day)
            # Apply shock multipliers on top
            shock_list = self.shock.step(intensity)
            shock_spread_mults = np.array([
                self.shock.get_spread_multiplier(i) for i in range(n)
            ])
            depth_fractions = np.array([
                self.shock.get_depth_fraction(i) for i in range(n)
            ])
            total_spread_mults = spread_mults * shock_spread_mults

            for m in total_spread_mults:
                self.metrics.record_spread_mult(float(m))

            if shock_list:
                self.metrics.record_shock()

            # Depth
            self.depth.step(n)

            # Adversary orders
            adversary_orders = self.adversary.step(mid_prices, spreads, intensity)
            self.metrics.record_adversary_orders(len(adversary_orders))

            # Latency
            latency_us = self.latency.sample_latency_us(intensity)
            self.metrics.record_latency(latency_us)

            # Kyle lambda
            if len(self.price_impact.current_lambda) > 0:
                self.metrics.record_kyle_lambda(float(self.price_impact.current_lambda.mean()))

            result = {
                "step": self._step,
                "intensity": intensity,
                "regime": self._current_regime,
                "regime_name": self._current_regime.name,
                "regime_changed": regime_changed,
                "regime_volatility": self.regime.get_volatility(),
                "regime_drift": self.regime.get_drift(),
                "spread_multipliers": total_spread_mults,
                "depth_fractions": depth_fractions,
                "latency_us": latency_us,
                "kyle_lambdas": self.price_impact.current_lambda.copy(),
                "adversary_orders": adversary_orders,
                "active_shocks": [s.shock_type.name for s in self.shock.active_shocks],
            }
            return result

    # ------------------------------------------------------------------
    # Augmentation helpers (for env integration)
    # ------------------------------------------------------------------

    def augment_spreads(
        self, spreads: np.ndarray, time_of_day: float = 0.5
    ) -> np.ndarray:
        """Apply spread noise + shocks to spreads."""
        n = len(spreads)
        mults = self.spread_noise.sample(n, time_of_day)
        shock_mults = np.array([self.shock.get_spread_multiplier(i) for i in range(n)])
        return self.spread_noise.apply_to_spread(spreads * shock_mults, time_of_day)

    def augment_observation(
        self,
        obs: np.ndarray,
        noise_std: float = 0.01,
        dropout_prob: float = 0.02,
    ) -> np.ndarray:
        """Apply observation noise and stale obs injection."""
        if not self.cfg.apply_to_observations:
            return obs
        intensity = self.annealer.intensity()
        augmented = self.augmentor.add_observation_noise(obs, noise_std, dropout_prob, intensity)
        augmented = self.augmentor.add_stale_observation(augmented, self._obs_history, 0.02, intensity)
        augmented = self.augmentor.apply_regime_feature_noise(
            augmented, self._current_regime, intensity
        )
        self._obs_history.append(obs.copy())
        if len(self._obs_history) > 50:
            self._obs_history.pop(0)
        return augmented

    def augment_action(
        self,
        action: np.ndarray,
        noise_scale: float = 0.005,
    ) -> np.ndarray:
        """Optionally add small action noise (simulates execution imprecision)."""
        if not self.cfg.apply_to_actions:
            return action
        intensity = self.annealer.intensity()
        noise = self.rng.normal(0.0, noise_scale * intensity, size=action.shape)
        return action + noise

    def sample_fill(
        self,
        order_id: int,
        order_size: float,
        queue_position: float = 0.5,
        spread: float = 0.01,
        mid_price: float = 100.0,
        limit_price: float = 100.0,
    ) -> Tuple[bool, float, int]:
        """Wrapper: sample fill outcome for an order."""
        intensity = self.annealer.intensity()
        filled, frac, delay = self.fill_rate.sample_fill(
            order_id, order_size, queue_position, spread, mid_price, limit_price, intensity
        )
        self.metrics.record_fill_rate(frac if filled else 0.0)
        return filled, frac, delay

    def compute_price_impact(
        self, trade_size: float, asset_id: int, sign: float = 1.0
    ) -> Tuple[float, float]:
        """Wrapper: compute (temp, perm) impact."""
        intensity = self.annealer.intensity()
        return self.price_impact.compute_impact(trade_size, asset_id, sign, intensity)

    def sample_latency(self) -> Tuple[float, bool, bool]:
        """
        Returns (latency_us, packet_dropped, resequenced).
        """
        intensity = self.annealer.intensity()
        lat = self.latency.sample_latency_us(intensity)
        dropped = self.latency.should_drop_packet()
        reseq = self.latency.should_resequence()
        return lat, dropped, reseq

    # ------------------------------------------------------------------
    # Curriculum interface
    # ------------------------------------------------------------------

    def record_episode_outcome(
        self,
        scenario_type: str,
        agent_pnl: float,
        benchmark_pnl: float = 0.0,
    ) -> None:
        """Record whether agent won/lost an episode for curriculum tracking."""
        win = agent_pnl > benchmark_pnl
        self.curriculum.record_outcome(scenario_type, win)

    def get_difficulty(self, scenario_type: Optional[str] = None) -> float:
        """Get current curriculum difficulty."""
        return self.curriculum.get_difficulty(scenario_type)

    # ------------------------------------------------------------------
    # Metrics / reporting
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        summary = self.metrics.get_summary()
        summary["curriculum"] = self.curriculum.get_curriculum_summary()
        summary["regime_history"] = self.regime.regime_history[-10:]
        summary["shock_history"] = self.shock.shock_history[-10:]
        summary["annealer_step"] = self.annealer.current_step
        summary["annealer_intensity"] = self.annealer.intensity()
        return summary

    # ------------------------------------------------------------------
    # Integration hooks for MultiAssetTradingEnv
    # ------------------------------------------------------------------

    def env_reset_hook(self, env: Any) -> None:
        """Call from env.reset() to inject randomization into env state."""
        if not self.cfg.reset_callback or not self.cfg.enabled:
            return
        state = self.reset()
        # Apply regime volatility to env
        if hasattr(env, "volatility"):
            env.volatility = np.full(self.cfg.num_assets, state["regime_volatility"])
        if hasattr(env, "price_drift"):
            env.price_drift = np.full(self.cfg.num_assets, state["regime_drift"])
        # Randomize initial spreads
        if hasattr(env, "spreads"):
            n = min(len(env.spreads), self.cfg.num_assets)
            mults = self.spread_noise.sample(n)
            env.spreads[:n] = env.spreads[:n] * mults
        # Randomize depth levels
        if hasattr(env, "order_book_levels"):
            env.order_book_levels = self.depth.num_levels

    def env_step_hook(
        self,
        env: Any,
        mid_prices: Optional[np.ndarray] = None,
        time_of_day: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Call from env.step() to get per-step randomization.

        Returns dict of randomization outputs that env should apply.
        """
        if not self.cfg.env_step_callback or not self.cfg.enabled:
            return {}

        if mid_prices is None:
            if hasattr(env, "mid_prices"):
                mid_prices = np.array(env.mid_prices)
            else:
                mid_prices = np.ones(self.cfg.num_assets) * 100.0

        if hasattr(env, "spreads"):
            spreads = np.array(env.spreads)
        else:
            spreads = np.ones(self.cfg.num_assets) * 0.01

        result = self.step(mid_prices, spreads, time_of_day)

        # Inject adversary orders into env order book
        if hasattr(env, "inject_orders") and result.get("adversary_orders"):
            for order in result["adversary_orders"]:
                try:
                    env.inject_orders([order])
                except Exception as e:
                    logger.debug("Failed to inject adversary order: %s", e)

        # Update env spreads
        if hasattr(env, "spreads"):
            n = min(len(env.spreads), self.cfg.num_assets)
            env.spreads[:n] = env.spreads[:n] * result["spread_multipliers"][:n]

        return result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config_dict(self) -> Dict[str, Any]:
        """Return DRE configuration as a plain dict."""
        import dataclasses
        return dataclasses.asdict(self.cfg)

    @classmethod
    def from_config_dict(cls, d: Dict[str, Any]) -> "DomainRandomizationEngine":
        """Reconstruct DRE from a config dict."""
        # Rebuild sub-configs
        cfg = DREConfig(
            spread_noise=SpreadNoiseConfig(**d.get("spread_noise", {})),
            fill_rate=FillRateConfig(**d.get("fill_rate", {})),
            latency=LatencyConfig(**d.get("latency", {})),
            price_impact=PriceImpactConfig(**d.get("price_impact", {})),
            order_book_depth=OrderBookDepthConfig(**d.get("order_book_depth", {})),
            liquidity_shock=LiquidityShockConfig(**d.get("liquidity_shock", {})),
            regime=RegimeConfig(**d.get("regime", {})),
            adversary=AdversaryConfig(**d.get("adversary", {})),
            curriculum=CurriculumConfig(**d.get("curriculum", {})),
            anneal=AnnealConfig(**{
                k: AnnealSchedule[v] if k == "schedule_type" else v
                for k, v in d.get("anneal", {}).items()
            }),
            **{k: v for k, v in d.items() if k not in (
                "spread_noise", "fill_rate", "latency", "price_impact",
                "order_book_depth", "liquidity_shock", "regime", "adversary",
                "curriculum", "anneal",
            )},
        )
        return cls(cfg)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_dre(
    num_assets: int = 4,
    num_agents: int = 8,
    seed: Optional[int] = None,
    difficulty: float = 0.5,
    enable_adversaries: bool = True,
    enable_shocks: bool = True,
    enable_regime: bool = True,
    anneal_steps: int = 500_000,
) -> DomainRandomizationEngine:
    """Create a DomainRandomizationEngine with common defaults."""
    cfg = DREConfig(
        num_assets=num_assets,
        num_agents=num_agents,
        seed=seed,
        adversary=AdversaryConfig(enabled=enable_adversaries),
        liquidity_shock=LiquidityShockConfig(enabled=enable_shocks),
        regime=RegimeConfig(enabled=enable_regime),
        anneal=AnnealConfig(
            total_steps=anneal_steps,
            initial_intensity=difficulty * 0.5,
            final_intensity=min(1.0, difficulty * 1.5),
        ),
        curriculum=CurriculumConfig(initial_difficulty=difficulty),
    )
    return DomainRandomizationEngine(cfg)


def make_light_dre(seed: Optional[int] = None) -> DomainRandomizationEngine:
    """Lightweight DRE for unit testing and quick experiments."""
    cfg = DREConfig(
        seed=seed,
        adversary=AdversaryConfig(enabled=False),
        liquidity_shock=LiquidityShockConfig(
            enabled=True,
            shock_prob_per_step=0.001,
        ),
        regime=RegimeConfig(enabled=True),
        anneal=AnnealConfig(schedule_type=AnnealSchedule.CONSTANT, initial_intensity=0.3),
    )
    return DomainRandomizationEngine(cfg)


def make_adversarial_dre(seed: Optional[int] = None) -> DomainRandomizationEngine:
    """Maximum-adversarial DRE for robustness training."""
    cfg = DREConfig(
        seed=seed,
        spread_noise=SpreadNoiseConfig(
            base_multiplier_std=0.8,
            max_multiplier=20.0,
        ),
        fill_rate=FillRateConfig(
            base_fill_prob=0.6,
            partial_fill_prob=0.7,
        ),
        latency=LatencyConfig(
            base_latency_us=500.0,
            network_congestion_prob=0.15,
            packet_loss_prob=0.01,
        ),
        price_impact=PriceImpactConfig(
            kyle_lambda_log_std=1.5,
            permanent_impact_fraction=0.5,
        ),
        liquidity_shock=LiquidityShockConfig(
            shock_prob_per_step=0.02,
            spread_widening_max_factor=100.0,
        ),
        adversary=AdversaryConfig(
            max_adversaries=10,
            predatory_hft_prob=0.3,
        ),
        anneal=AnnealConfig(
            schedule_type=AnnealSchedule.CONSTANT,
            initial_intensity=1.0,
        ),
    )
    return DomainRandomizationEngine(cfg)


# ---------------------------------------------------------------------------
# Utility: scenario sampler for curriculum
# ---------------------------------------------------------------------------

class ScenarioSampler:
    """
    Samples training scenarios for the auto-curriculum.

    Each scenario specifies which DRE components are active and at what intensity.
    """

    SCENARIO_REGISTRY: Dict[str, Dict[str, Any]] = {
        "baseline": {
            "description": "Standard sim environment, minimal noise",
            "spread_noise_intensity": 0.1,
            "fill_noise": False,
            "latency_noise": False,
            "price_impact_noise": False,
            "liquidity_shock": False,
            "regime": False,
            "adversaries": False,
        },
        "spread_noise_light": {
            "description": "Light spread noise only",
            "spread_noise_intensity": 0.3,
            "fill_noise": False,
            "latency_noise": False,
            "price_impact_noise": False,
            "liquidity_shock": False,
            "regime": False,
            "adversaries": False,
        },
        "spread_noise_heavy": {
            "description": "Heavy spread noise",
            "spread_noise_intensity": 1.0,
            "fill_noise": False,
            "latency_noise": False,
            "price_impact_noise": False,
            "liquidity_shock": False,
            "regime": False,
            "adversaries": False,
        },
        "fill_randomization": {
            "description": "Partial fill randomization",
            "spread_noise_intensity": 0.3,
            "fill_noise": True,
            "latency_noise": False,
            "price_impact_noise": False,
            "liquidity_shock": False,
            "regime": False,
            "adversaries": False,
        },
        "latency": {
            "description": "Latency jitter",
            "spread_noise_intensity": 0.2,
            "fill_noise": False,
            "latency_noise": True,
            "price_impact_noise": False,
            "liquidity_shock": False,
            "regime": False,
            "adversaries": False,
        },
        "price_impact": {
            "description": "Kyle lambda randomization",
            "spread_noise_intensity": 0.2,
            "fill_noise": False,
            "latency_noise": False,
            "price_impact_noise": True,
            "liquidity_shock": False,
            "regime": False,
            "adversaries": False,
        },
        "liquidity_shock": {
            "description": "Liquidity shock injection",
            "spread_noise_intensity": 0.3,
            "fill_noise": True,
            "latency_noise": True,
            "price_impact_noise": True,
            "liquidity_shock": True,
            "regime": False,
            "adversaries": False,
        },
        "regime_switch": {
            "description": "Regime transitions",
            "spread_noise_intensity": 0.4,
            "fill_noise": True,
            "latency_noise": False,
            "price_impact_noise": True,
            "liquidity_shock": False,
            "regime": True,
            "adversaries": False,
        },
        "adversary": {
            "description": "Adversarial participants",
            "spread_noise_intensity": 0.3,
            "fill_noise": True,
            "latency_noise": True,
            "price_impact_noise": True,
            "liquidity_shock": False,
            "regime": True,
            "adversaries": True,
        },
        "full_chaos": {
            "description": "All randomization at maximum",
            "spread_noise_intensity": 1.0,
            "fill_noise": True,
            "latency_noise": True,
            "price_impact_noise": True,
            "liquidity_shock": True,
            "regime": True,
            "adversaries": True,
        },
    }

    def __init__(
        self,
        curriculum: AutoCurriculum,
        rng: np.random.Generator,
    ) -> None:
        self.curriculum = curriculum
        self.rng = rng

    def sample_scenario(self) -> Tuple[str, Dict[str, Any]]:
        """Sample a training scenario weighted by curriculum difficulty."""
        names = list(self.SCENARIO_REGISTRY.keys())
        # Weight easier scenarios more at low difficulty
        global_diff = self.curriculum.get_difficulty()
        n = len(names)
        weights = np.zeros(n)
        for i, name in enumerate(names):
            scenario_diff = i / max(n - 1, 1)
            # Gaussian centered around global_diff
            distance = abs(scenario_diff - global_diff)
            weights[i] = math.exp(-4.0 * distance ** 2)
        weights /= weights.sum()
        idx = int(self.rng.choice(n, p=weights))
        name = names[idx]
        return name, self.SCENARIO_REGISTRY[name].copy()

    def build_dre_for_scenario(
        self,
        scenario_name: str,
        num_assets: int = 4,
        seed: Optional[int] = None,
    ) -> DomainRandomizationEngine:
        """Build a DRE configured for a specific scenario."""
        spec = self.SCENARIO_REGISTRY.get(scenario_name, {})
        cfg = DREConfig(
            num_assets=num_assets,
            seed=seed,
            spread_noise=SpreadNoiseConfig(
                enabled=True,
                base_multiplier_std=spec.get("spread_noise_intensity", 0.3),
            ),
            fill_rate=FillRateConfig(enabled=bool(spec.get("fill_noise", False))),
            latency=LatencyConfig(enabled=bool(spec.get("latency_noise", False))),
            price_impact=PriceImpactConfig(enabled=bool(spec.get("price_impact_noise", False))),
            liquidity_shock=LiquidityShockConfig(enabled=bool(spec.get("liquidity_shock", False))),
            regime=RegimeConfig(enabled=bool(spec.get("regime", False))),
            adversary=AdversaryConfig(enabled=bool(spec.get("adversaries", False))),
        )
        return DomainRandomizationEngine(cfg)


# ---------------------------------------------------------------------------
# Parameter schedule utilities
# ---------------------------------------------------------------------------

class ParameterSchedule:
    """
    Utility class to schedule a single parameter value over training.

    Useful for fine-grained control of individual randomization intensities.
    """

    def __init__(
        self,
        values: List[float],
        milestones: List[int],
        interpolation: str = "linear",
    ) -> None:
        assert len(values) == len(milestones) + 1, (
            "len(values) must equal len(milestones) + 1"
        )
        self.values = values
        self.milestones = milestones
        self.interpolation = interpolation
        self._step = 0

    def step(self) -> float:
        self._step += 1
        return self.get()

    def get(self) -> float:
        t = self._step
        # Find segment
        for i, m in enumerate(self.milestones):
            if t < m:
                if i == 0:
                    return self.values[0]
                t0 = self.milestones[i - 1] if i > 0 else 0
                t1 = m
                v0 = self.values[i]
                v1 = self.values[i + 1]
                frac = (t - t0) / max(t1 - t0, 1)
                if self.interpolation == "linear":
                    return v0 + frac * (v1 - v0)
                elif self.interpolation == "cosine":
                    return v1 + 0.5 * (v0 - v1) * (1.0 + math.cos(math.pi * frac))
                else:
                    return v0 + frac * (v1 - v0)
        return self.values[-1]

    def reset(self) -> None:
        self._step = 0

    @property
    def current_step(self) -> int:
        return self._step


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "MarketRegime",
    "AdversaryType",
    "AnnealSchedule",
    "LiquidityShockType",
    # Configs
    "DREConfig",
    "SpreadNoiseConfig",
    "FillRateConfig",
    "LatencyConfig",
    "PriceImpactConfig",
    "OrderBookDepthConfig",
    "LiquidityShockConfig",
    "RegimeConfig",
    "AdversaryConfig",
    "CurriculumConfig",
    "AnnealConfig",
    # Sub-modules
    "SpreadNoiseGenerator",
    "FillRateRandomizer",
    "LatencyJitterInjector",
    "PriceImpactRandomizer",
    "OrderBookDepthRandomizer",
    "LiquidityShockInjector",
    "LiquidityShockState",
    "RegimeManager",
    "AdversarialParticipantInjector",
    "AdversaryState",
    "AutoCurriculum",
    "AnnealScheduler",
    "ObservationAugmentor",
    "DREMetricsTracker",
    # Main engine
    "DomainRandomizationEngine",
    # Utilities
    "ScenarioSampler",
    "ParameterSchedule",
    # Factories
    "make_dre",
    "make_light_dre",
    "make_adversarial_dre",
    # Extended
    "MarketMicrostructureNoise",
    "CrossAssetContagionModel",
    "SeasonalityModel",
    "OrderFlowToxicityInjector",
    "VolatilityClusteringSimulator",
    "IntraEpisodeShockSequencer",
    "RandomizationLogger",
    "DRECheckpointer",
    "MultiEpisodeCurriculumTracker",
    "RegimeCalibrator",
]


# ---------------------------------------------------------------------------
# Extended: MarketMicrostructureNoise
# ---------------------------------------------------------------------------

class MarketMicrostructureNoise:
    """
    Comprehensive market microstructure noise model.

    Models various noise sources that occur in real markets but are absent
    or simplified in simulation:
    - Trade-through noise: orders occasionally executed at non-best prices
    - Tick constraint noise: prices constrained to discrete tick grid
    - Quote stuffing noise: sudden burst of quotes followed by cancellations
    - Last-look noise: market makers can reject orders after seeing them
    - Settlement noise: small end-of-day price adjustments
    """

    def __init__(
        self,
        tick_size: float = 0.01,
        trade_through_prob: float = 0.02,
        quote_stuffing_prob: float = 0.005,
        last_look_reject_prob: float = 0.03,
        settlement_noise_std: float = 0.001,
        seed: Optional[int] = None,
    ) -> None:
        self.tick_size = tick_size
        self.trade_through_prob = trade_through_prob
        self.quote_stuffing_prob = quote_stuffing_prob
        self.last_look_reject_prob = last_look_reject_prob
        self.settlement_noise_std = settlement_noise_std
        self.rng = np.random.default_rng(seed)
        self._quote_stuffing_active = False
        self._stuffing_duration = 0
        self._step = 0

    def apply_tick_constraint(self, price: float) -> float:
        """Round price to nearest tick."""
        return round(price / self.tick_size) * self.tick_size

    def apply_trade_through_noise(
        self, execution_price: float, mid_price: float, spread: float
    ) -> float:
        """Occasionally execute at slightly worse price (trade-through)."""
        if self.rng.random() < self.trade_through_prob:
            # Execute at slightly worse price
            slippage = self.rng.uniform(0, spread * 0.5)
            direction = 1 if execution_price >= mid_price else -1
            execution_price += direction * slippage
        return self.apply_tick_constraint(execution_price)

    def apply_last_look(self, order_price: float, mid_price: float) -> bool:
        """
        Simulate last-look rejection.

        Returns True if order is accepted, False if rejected.
        """
        if self.rng.random() < self.last_look_reject_prob:
            # Likely to reject if order is aggressive (arb against MM)
            spread_fraction = abs(order_price - mid_price) / max(mid_price * 0.001, 1e-8)
            if spread_fraction < 0.5:
                return False
        return True

    def apply_quote_stuffing(
        self, observable_quotes: int
    ) -> Tuple[int, bool]:
        """
        Simulate quote stuffing: sudden burst of phantom quotes.

        Returns (actual_quotes_to_process, is_stuffing_active).
        """
        self._step += 1
        if self._quote_stuffing_active:
            self._stuffing_duration -= 1
            if self._stuffing_duration <= 0:
                self._quote_stuffing_active = False
            # During stuffing, overwhelm with fake quotes
            fake_quotes = int(self.rng.integers(100, 500))
            return observable_quotes + fake_quotes, True

        if self.rng.random() < self.quote_stuffing_prob:
            self._quote_stuffing_active = True
            self._stuffing_duration = int(self.rng.integers(3, 20))

        return observable_quotes, False

    def apply_settlement_noise(self, closing_price: float) -> float:
        """Add small settlement adjustment."""
        noise = self.rng.normal(0, self.settlement_noise_std * closing_price)
        return self.apply_tick_constraint(closing_price + noise)

    def apply_all(
        self,
        price: float,
        mid_price: float,
        spread: float,
        is_order: bool = True,
    ) -> Tuple[float, bool]:
        """
        Apply all noise models.

        Returns (noisy_price, order_accepted).
        """
        if is_order:
            accepted = self.apply_last_look(price, mid_price)
            if not accepted:
                return price, False
        price = self.apply_trade_through_noise(price, mid_price, spread)
        return price, True


# ---------------------------------------------------------------------------
# Extended: CrossAssetContagionModel
# ---------------------------------------------------------------------------

class CrossAssetContagionModel:
    """
    Models cross-asset contagion during liquidity crises.

    During a stress event, correlations spike and liquidity dries up
    simultaneously across multiple assets. This is a key sim-to-real gap
    that naive domain randomization misses.

    Implements:
    - Dynamic conditional correlation (DCC) driven contagion
    - Jump contagion: when one asset has a large move, others follow
    - Liquidity contagion: when one asset's depth drops, others drop too
    - Flight-to-quality: some assets benefit while others suffer
    """

    def __init__(
        self,
        num_assets: int = 4,
        base_correlation: float = 0.3,
        crisis_correlation: float = 0.85,
        jump_contagion_prob: float = 0.3,
        jump_contagion_scale: float = 0.5,
        liquidity_contagion_factor: float = 0.4,
        flight_to_quality_assets: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.num_assets = num_assets
        self.base_correlation = base_correlation
        self.crisis_correlation = crisis_correlation
        self.jump_contagion_prob = jump_contagion_prob
        self.jump_contagion_scale = jump_contagion_scale
        self.liquidity_contagion_factor = liquidity_contagion_factor
        self.flight_to_quality = flight_to_quality_assets or []
        self.rng = np.random.default_rng(seed)

        # State
        self._crisis_intensity: float = 0.0
        self._correlation_matrix = self._build_correlation_matrix(base_correlation)
        self._depth_multipliers = np.ones(num_assets)
        self._spread_multipliers = np.ones(num_assets)

    def _build_correlation_matrix(self, corr: float) -> np.ndarray:
        n = self.num_assets
        C = np.full((n, n), corr)
        np.fill_diagonal(C, 1.0)
        return C

    def update(
        self,
        returns: np.ndarray,
        current_spreads: np.ndarray,
        current_depths: np.ndarray,
        regime: MarketRegime,
    ) -> Dict[str, Any]:
        """
        Update contagion state based on current market conditions.

        Returns contagion effects for each asset.
        """
        # Measure shock magnitude
        max_abs_return = float(np.max(np.abs(returns)))

        # Crisis intensity driven by returns
        crisis_trigger = max_abs_return > 0.02
        if crisis_trigger:
            self._crisis_intensity = min(1.0, self._crisis_intensity + 0.2)
        else:
            self._crisis_intensity *= 0.95

        # Regime also affects crisis
        if regime in (MarketRegime.CRASH, MarketRegime.CRISIS):
            self._crisis_intensity = min(1.0, self._crisis_intensity + 0.1)

        # Update correlation matrix
        target_corr = (
            self.base_correlation
            + self._crisis_intensity * (self.crisis_correlation - self.base_correlation)
        )
        self._correlation_matrix = self._build_correlation_matrix(target_corr)

        # Jump contagion: if any asset jumps, others may too
        contagion_returns = returns.copy()
        for i in range(self.num_assets):
            if abs(returns[i]) > 0.03:
                if self.rng.random() < self.jump_contagion_prob:
                    for j in range(self.num_assets):
                        if j != i and j not in self.flight_to_quality:
                            contagion_returns[j] += (
                                returns[i] * self.jump_contagion_scale
                                * self.rng.uniform(0.2, 0.8)
                            )

        # Flight to quality: some assets go up when others crash
        for j in self.flight_to_quality:
            if j < len(contagion_returns):
                # Opposite effect during crisis
                avg_other = np.mean([
                    contagion_returns[k] for k in range(self.num_assets)
                    if k != j
                ])
                if avg_other < -0.01:
                    contagion_returns[j] -= avg_other * 0.3

        # Liquidity contagion
        min_depth_fraction = float(np.min(
            np.maximum(current_depths / (np.maximum(current_depths.mean(), 1.0)), 0.1)
        ))
        if min_depth_fraction < 0.5:
            contagion_factor = 1.0 - self.liquidity_contagion_factor * (1 - min_depth_fraction)
            self._depth_multipliers = np.maximum(
                0.05, np.ones(self.num_assets) * contagion_factor
            )
        else:
            self._depth_multipliers = np.ones(self.num_assets)

        # Spread contagion
        spread_contagion = 1.0 + self._crisis_intensity * 3.0
        self._spread_multipliers = np.full(self.num_assets, spread_contagion)

        return {
            "contagion_returns": contagion_returns,
            "correlation_matrix": self._correlation_matrix.copy(),
            "depth_multipliers": self._depth_multipliers.copy(),
            "spread_multipliers": self._spread_multipliers.copy(),
            "crisis_intensity": self._crisis_intensity,
        }

    def sample_correlated_returns(
        self, individual_vols: np.ndarray
    ) -> np.ndarray:
        """Sample correlated returns using current correlation matrix."""
        try:
            L = np.linalg.cholesky(
                self._correlation_matrix + np.eye(self.num_assets) * 1e-6
            )
        except np.linalg.LinAlgError:
            L = np.eye(self.num_assets)
        z = self.rng.standard_normal(self.num_assets)
        corr_z = L @ z
        return individual_vols * corr_z

    @property
    def crisis_intensity(self) -> float:
        return self._crisis_intensity

    @property
    def current_correlation(self) -> np.ndarray:
        return self._correlation_matrix.copy()


# ---------------------------------------------------------------------------
# Extended: SeasonalityModel
# ---------------------------------------------------------------------------

class SeasonalityModel:
    """
    Models intraday and weekly seasonality in market microstructure.

    Real markets exhibit strong time-of-day patterns:
    - Open: high volume, high spread, high volatility
    - Mid-morning: spread narrows, vol decreases
    - Lunch: low vol, low volume
    - Afternoon: increasing vol toward close
    - Close: highest volume, U-shape spread
    - Monday: typically lower vol
    - Friday: higher vol going into weekend
    """

    # Intraday spread multiplier (30-minute buckets, 13 buckets for 6.5hr day)
    DEFAULT_SPREAD_PATTERN = np.array([
        2.5,   # 9:30 open
        1.8,   # 10:00
        1.4,   # 10:30
        1.2,   # 11:00
        1.1,   # 11:30
        1.0,   # 12:00 midday
        1.0,   # 12:30
        1.1,   # 13:00
        1.1,   # 13:30
        1.2,   # 14:00
        1.4,   # 14:30
        1.8,   # 15:00
        2.2,   # 15:30
    ])

    DEFAULT_VOL_PATTERN = np.array([
        3.0,   # 9:30 open
        2.0,   # 10:00
        1.5,   # 10:30
        1.2,   # 11:00
        1.1,   # 11:30
        1.0,   # 12:00 midday
        1.0,   # 12:30
        1.1,   # 13:00
        1.1,   # 13:30
        1.2,   # 14:00
        1.4,   # 14:30
        1.8,   # 15:00
        2.5,   # 15:30
    ])

    WEEKLY_VOL_PATTERN = np.array([1.05, 1.0, 1.0, 1.0, 1.1])  # Mon-Fri

    def __init__(
        self,
        spread_pattern: Optional[np.ndarray] = None,
        vol_pattern: Optional[np.ndarray] = None,
        weekly_pattern: Optional[np.ndarray] = None,
        noise_std: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self.spread_pattern = spread_pattern if spread_pattern is not None else self.DEFAULT_SPREAD_PATTERN.copy()
        self.vol_pattern = vol_pattern if vol_pattern is not None else self.DEFAULT_VOL_PATTERN.copy()
        self.weekly_pattern = weekly_pattern if weekly_pattern is not None else self.WEEKLY_VOL_PATTERN.copy()
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def get_spread_multiplier(self, time_of_day: float, day_of_week: int = 2) -> float:
        """
        Get spread multiplier for given time.

        Args:
            time_of_day: 0=open, 1=close
            day_of_week: 0=Monday, 4=Friday
        """
        n = len(self.spread_pattern)
        idx = min(int(time_of_day * n), n - 1)
        base = float(self.spread_pattern[idx])
        dow_adj = self.weekly_pattern[day_of_week % len(self.weekly_pattern)]
        noise = self.rng.normal(0, self.noise_std)
        return max(0.5, base * dow_adj * (1 + noise))

    def get_vol_multiplier(self, time_of_day: float, day_of_week: int = 2) -> float:
        """Get volatility multiplier for given time."""
        n = len(self.vol_pattern)
        idx = min(int(time_of_day * n), n - 1)
        base = float(self.vol_pattern[idx])
        dow_adj = self.weekly_pattern[day_of_week % len(self.weekly_pattern)]
        noise = self.rng.normal(0, self.noise_std)
        return max(0.3, base * dow_adj * (1 + noise))

    def get_volume_multiplier(self, time_of_day: float) -> float:
        """Get volume multiplier (U-shaped through day)."""
        # U-shape: high at start and end, low at midday
        t = time_of_day
        vol_mult = 2.0 * (2.0 * (t - 0.5) ** 2 + 0.5)
        noise = self.rng.normal(0, self.noise_std)
        return max(0.2, vol_mult * (1 + noise))

    def interpolate(self, time_of_day: float, pattern: np.ndarray) -> float:
        """Smooth interpolation within a pattern."""
        n = len(pattern)
        frac = time_of_day * (n - 1)
        i = int(frac)
        i = min(i, n - 2)
        w = frac - i
        return float(pattern[i] * (1 - w) + pattern[i + 1] * w)


# ---------------------------------------------------------------------------
# Extended: OrderFlowToxicityInjector
# ---------------------------------------------------------------------------

class OrderFlowToxicityInjector:
    """
    Injects toxic order flow patterns into the training environment.

    Real markets have episodes of toxic order flow that adverse-select
    market makers. This module simulates these patterns:
    - PIN (probability of informed trading) spikes
    - VPIN (volume-synchronized PIN) events
    - Toxic news-driven flow
    - Institutional block trading that fragments across venues
    """

    def __init__(
        self,
        base_pin: float = 0.1,
        max_pin: float = 0.8,
        toxicity_event_prob: float = 0.01,
        toxicity_event_duration: int = 30,
        institutional_flow_prob: float = 0.005,
        institutional_flow_size: float = 1000.0,
        seed: Optional[int] = None,
    ) -> None:
        self.base_pin = base_pin
        self.max_pin = max_pin
        self.toxicity_event_prob = toxicity_event_prob
        self.toxicity_event_duration = toxicity_event_duration
        self.institutional_flow_prob = institutional_flow_prob
        self.institutional_flow_size = institutional_flow_size
        self.rng = np.random.default_rng(seed)

        self._current_pin: float = base_pin
        self._toxicity_event_active: bool = False
        self._toxicity_steps_remaining: int = 0
        self._institutional_direction: float = 0.0
        self._institutional_remaining: float = 0.0
        self._step: int = 0
        self._vpin_history: collections.deque = collections.deque(maxlen=50)

    def step(self, volume: float) -> Dict[str, Any]:
        """
        Advance one step. Returns toxicity metrics for this step.
        """
        self._step += 1

        # Update toxicity event
        if self._toxicity_event_active:
            self._toxicity_steps_remaining -= 1
            if self._toxicity_steps_remaining <= 0:
                self._toxicity_event_active = False
                self._current_pin = self.base_pin
        elif self.rng.random() < self.toxicity_event_prob:
            self._toxicity_event_active = True
            self._toxicity_steps_remaining = self.toxicity_event_duration
            self._current_pin = self.rng.uniform(0.4, self.max_pin)

        # VPIN calculation (simplified)
        buy_vol = volume * (0.5 + self._current_pin * self.rng.normal(0.1, 0.05))
        sell_vol = volume - buy_vol
        vpin = abs(buy_vol - sell_vol) / max(volume, 1e-8)
        self._vpin_history.append(vpin)

        # Institutional flow injection
        institutional_orders: List[Dict[str, Any]] = []
        if self._institutional_remaining > 0:
            fragment = min(self._institutional_remaining, self.institutional_flow_size * 0.1)
            institutional_orders.append({
                "side": "buy" if self._institutional_direction > 0 else "sell",
                "size": fragment,
                "type": "market",
                "is_institutional": True,
            })
            self._institutional_remaining -= fragment
        elif self.rng.random() < self.institutional_flow_prob:
            self._institutional_direction = 1.0 if self.rng.random() < 0.5 else -1.0
            self._institutional_remaining = self.institutional_flow_size

        return {
            "pin": self._current_pin,
            "vpin": vpin,
            "vpin_ema": float(np.mean(list(self._vpin_history))) if self._vpin_history else vpin,
            "toxicity_event_active": self._toxicity_event_active,
            "institutional_orders": institutional_orders,
        }

    def get_adverse_selection_cost(self, spread: float) -> float:
        """Estimate adverse selection cost as fraction of spread."""
        return spread * self._current_pin * 0.5

    @property
    def current_pin(self) -> float:
        return self._current_pin

    @property
    def vpin_ema(self) -> float:
        if not self._vpin_history:
            return 0.0
        return float(np.mean(list(self._vpin_history)))


# ---------------------------------------------------------------------------
# Extended: VolatilityClusteringSimulator
# ---------------------------------------------------------------------------

class VolatilityClusteringSimulator:
    """
    Simulates volatility clustering (GARCH-like) for more realistic price dynamics.

    Real markets exhibit strong volatility clustering: high vol follows high vol.
    Simple random volatility misses this. This module provides GARCH(1,1) simulation
    with regime-switching to improve sim fidelity.
    """

    def __init__(
        self,
        num_assets: int = 4,
        omega: float = 1e-6,
        alpha: float = 0.1,
        beta: float = 0.85,
        initial_vol: float = 0.01,
        use_regime_switching: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.num_assets = num_assets
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.initial_vol = initial_vol
        self.use_regime_switching = use_regime_switching
        self.rng = np.random.default_rng(seed)

        self._variances = np.full(num_assets, initial_vol ** 2)
        self._vols = np.full(num_assets, initial_vol)
        self._last_returns = np.zeros(num_assets)
        self._regime_vols = np.array([initial_vol * 0.5, initial_vol, initial_vol * 3.0])
        self._regime_probs = np.array([0.3, 0.5, 0.2])
        self._current_regime = 1

    def reset(self, initial_vol: Optional[float] = None) -> None:
        vol = initial_vol or self.initial_vol
        self._variances = np.full(self.num_assets, vol ** 2)
        self._vols = np.full(self.num_assets, vol)
        self._last_returns = np.zeros(self.num_assets)

    def step(self, returns: np.ndarray) -> np.ndarray:
        """
        Update GARCH state and return next-step volatility estimates.

        Args:
            returns: realized returns at this step (shape: num_assets)

        Returns:
            updated volatility estimates (shape: num_assets)
        """
        # GARCH(1,1) update
        self._variances = (
            self.omega
            + self.alpha * returns ** 2
            + self.beta * self._variances
        )
        self._vols = np.sqrt(np.maximum(self._variances, 1e-12))

        # Regime switching
        if self.use_regime_switching:
            if self.rng.random() < 0.02:  # 2% per step transition
                self._current_regime = int(
                    self.rng.choice(3, p=self._regime_probs)
                )
                regime_scale = self._regime_vols[self._current_regime] / max(
                    self.initial_vol, 1e-10
                )
                self._vols = np.clip(self._vols * regime_scale, 1e-6, 0.5)
                self._variances = self._vols ** 2

        self._last_returns = returns.copy()
        return self._vols.copy()

    def sample_returns(self) -> np.ndarray:
        """Sample next-step returns from current GARCH state."""
        return self.rng.standard_normal(self.num_assets) * self._vols

    @property
    def current_vols(self) -> np.ndarray:
        return self._vols.copy()

    @property
    def long_run_vol(self) -> float:
        """Unconditional long-run volatility from GARCH parameters."""
        return math.sqrt(self.omega / max(1 - self.alpha - self.beta, 1e-10))


# ---------------------------------------------------------------------------
# Extended: IntraEpisodeShockSequencer
# ---------------------------------------------------------------------------

class IntraEpisodeShockSequencer:
    """
    Orchestrates a sequence of shocks within a single training episode.

    Rather than random independent shocks, this creates realistic shock
    sequences: a large shock is often followed by elevated vol, then
    gradual recovery, occasionally with aftershocks.

    Shock sequence types:
    - Flash crash + recovery
    - Liquidity spiral: spread → withdrawal → more spread → crash
    - Contagion cascade: one asset → neighboring assets
    - Regime drift: slow regime transition over episode
    """

    SEQUENCE_TYPES = ["flash_crash", "liquidity_spiral", "contagion_cascade", "regime_drift", "none"]

    def __init__(
        self,
        num_assets: int = 4,
        sequence_prob: float = 0.3,
        seed: Optional[int] = None,
    ) -> None:
        self.num_assets = num_assets
        self.sequence_prob = sequence_prob
        self.rng = np.random.default_rng(seed)
        self._active_sequence: Optional[str] = None
        self._sequence_step: int = 0
        self._sequence_duration: int = 0
        self._sequence_params: Dict[str, Any] = {}
        self._step: int = 0

    def reset(self) -> None:
        self._active_sequence = None
        self._sequence_step = 0
        self._step = 0
        if self.rng.random() < self.sequence_prob:
            self._start_sequence()

    def _start_sequence(self) -> None:
        seq_type = self.rng.choice(self.SEQUENCE_TYPES[:-1])  # exclude "none"
        self._active_sequence = seq_type
        self._sequence_step = 0

        if seq_type == "flash_crash":
            self._sequence_duration = int(self.rng.integers(20, 60))
            self._sequence_params = {
                "crash_start": int(self.rng.integers(50, 150)),
                "crash_magnitude": float(self.rng.uniform(0.03, 0.10)),
                "recovery_steps": int(self.rng.integers(20, 50)),
                "primary_asset": int(self.rng.integers(0, self.num_assets)),
            }
        elif seq_type == "liquidity_spiral":
            self._sequence_duration = int(self.rng.integers(50, 150))
            self._sequence_params = {
                "initial_spread_mult": float(self.rng.uniform(2.0, 5.0)),
                "depth_drain_rate": float(self.rng.uniform(0.02, 0.05)),
                "spiral_assets": list(self.rng.choice(
                    self.num_assets,
                    size=min(3, self.num_assets),
                    replace=False,
                )),
            }
        elif seq_type == "contagion_cascade":
            self._sequence_duration = int(self.rng.integers(30, 100))
            self._sequence_params = {
                "trigger_asset": int(self.rng.integers(0, self.num_assets)),
                "cascade_delay": int(self.rng.integers(5, 15)),
                "contagion_strength": float(self.rng.uniform(0.3, 0.8)),
            }
        elif seq_type == "regime_drift":
            self._sequence_duration = int(self.rng.integers(100, 300))
            self._sequence_params = {
                "target_regime": self.rng.choice([
                    "VOLATILE", "CRASH", "CRISIS", "ILLIQUID"
                ]),
                "drift_rate": float(self.rng.uniform(0.005, 0.02)),
            }

    def step(self) -> Dict[str, Any]:
        """Advance one step and return shock parameters for this step."""
        self._step += 1
        result: Dict[str, Any] = {
            "sequence_active": self._active_sequence is not None,
            "sequence_type": self._active_sequence or "none",
            "spread_multipliers": np.ones(self.num_assets),
            "depth_multipliers": np.ones(self.num_assets),
            "price_shocks": np.zeros(self.num_assets),
        }

        if self._active_sequence is None:
            return result

        self._sequence_step += 1
        params = self._sequence_params

        if self._active_sequence == "flash_crash":
            crash_start = params.get("crash_start", 50)
            mag = params.get("crash_magnitude", 0.05)
            rec = params.get("recovery_steps", 30)
            asset = params.get("primary_asset", 0)

            if self._sequence_step == crash_start:
                result["price_shocks"][asset] = -mag
                result["spread_multipliers"] = np.full(self.num_assets, 10.0)
            elif crash_start < self._sequence_step < crash_start + rec:
                # Recovery
                recovery_frac = (self._sequence_step - crash_start) / rec
                result["price_shocks"][asset] = mag * 0.7 * (recovery_frac / rec)
                spread_mult = 10.0 * (1 - recovery_frac * 0.8)
                result["spread_multipliers"] = np.full(self.num_assets, max(1.0, spread_mult))

        elif self._active_sequence == "liquidity_spiral":
            spiral_assets = params.get("spiral_assets", [0])
            init_mult = params.get("initial_spread_mult", 3.0)
            drain_rate = params.get("depth_drain_rate", 0.03)
            prog = min(1.0, self._sequence_step / max(self._sequence_duration, 1))

            spread_mult = init_mult + prog * 5.0  # worsening
            depth_mult = max(0.05, 1.0 - drain_rate * self._sequence_step)

            for a in spiral_assets:
                if a < self.num_assets:
                    result["spread_multipliers"][a] = spread_mult
                    result["depth_multipliers"][a] = depth_mult

        elif self._active_sequence == "contagion_cascade":
            trigger = params.get("trigger_asset", 0)
            delay = params.get("cascade_delay", 10)
            strength = params.get("contagion_strength", 0.5)

            if self._sequence_step == 1:
                result["price_shocks"][trigger] = -0.03
                result["spread_multipliers"][trigger] = 5.0

            for a in range(self.num_assets):
                if a != trigger:
                    cascade_step = self._sequence_step - delay * (a + 1)
                    if 0 < cascade_step < 20:
                        result["price_shocks"][a] = -0.03 * strength * (1 - cascade_step / 20)
                        result["spread_multipliers"][a] = 1 + 4 * strength * (1 - cascade_step / 20)

        # Check if sequence ended
        if self._sequence_step >= self._sequence_duration:
            self._active_sequence = None
            self._sequence_step = 0

        return result

    @property
    def active_sequence(self) -> Optional[str]:
        return self._active_sequence


# ---------------------------------------------------------------------------
# Extended: RandomizationLogger
# ---------------------------------------------------------------------------

class RandomizationLogger:
    """
    Structured logger for all randomization events.

    Stores events in a queryable format for analysis and debugging.
    """

    def __init__(self, max_events: int = 10_000) -> None:
        self._events: collections.deque = collections.deque(maxlen=max_events)
        self._event_counts: Dict[str, int] = collections.defaultdict(int)
        self._step: int = 0

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        self._step += 1
        self._events.append({
            "step": self._step,
            "event_type": event_type,
            **payload,
        })
        self._event_counts[event_type] += 1

    def query(self, event_type: str) -> List[Dict[str, Any]]:
        return [e for e in self._events if e["event_type"] == event_type]

    def get_counts(self) -> Dict[str, int]:
        return dict(self._event_counts)

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self._events)

    def clear(self) -> None:
        self._events.clear()
        self._event_counts.clear()

    @property
    def num_events(self) -> int:
        return len(self._events)


# ---------------------------------------------------------------------------
# Extended: DRECheckpointer
# ---------------------------------------------------------------------------

class DRECheckpointer:
    """
    Saves and loads DRE state for reproducibility and experiment management.
    """

    def __init__(self, checkpoint_dir: str = "/tmp/dre_checkpoints") -> None:
        self.checkpoint_dir = checkpoint_dir
        self._checkpoints: Dict[str, Dict[str, Any]] = {}

    def save(self, dre: DomainRandomizationEngine, name: str) -> None:
        """Save DRE config and curriculum state."""
        state = {
            "config": dre.get_config_dict(),
            "curriculum": dre.curriculum.get_curriculum_summary(),
            "anneal_step": dre.annealer.current_step,
            "regime_history": dre.regime.regime_history[-20:],
            "metrics": dre.get_metrics(),
        }
        self._checkpoints[name] = state

    def load_config(self, name: str) -> Optional[DREConfig]:
        """Load DRE config from checkpoint."""
        if name not in self._checkpoints:
            return None
        return self._checkpoints[name].get("config")

    def list_checkpoints(self) -> List[str]:
        return list(self._checkpoints.keys())

    def delete(self, name: str) -> bool:
        if name in self._checkpoints:
            del self._checkpoints[name]
            return True
        return False


# ---------------------------------------------------------------------------
# Extended: MultiEpisodeCurriculumTracker
# ---------------------------------------------------------------------------

class MultiEpisodeCurriculumTracker:
    """
    Tracks agent performance across multiple training episodes for curriculum.

    Maintains a rolling window of episode outcomes and adjusts scenario
    difficulty at the episode level (rather than step level).
    """

    def __init__(
        self,
        num_scenario_types: int = 10,
        window: int = 100,
        difficulty_update_interval: int = 10,
    ) -> None:
        self.num_scenario_types = num_scenario_types
        self.window = window
        self.difficulty_update_interval = difficulty_update_interval

        self._episode_returns: Dict[int, collections.deque] = {
            i: collections.deque(maxlen=window) for i in range(num_scenario_types)
        }
        self._difficulties: np.ndarray = np.full(num_scenario_types, 0.3)
        self._episode_count: int = 0
        self._scenario_counts: np.ndarray = np.zeros(num_scenario_types, dtype=int)

    def record_episode(
        self, scenario_type: int, episode_return: float, baseline_return: float = 0.0
    ) -> None:
        if 0 <= scenario_type < self.num_scenario_types:
            normalized = episode_return - baseline_return
            self._episode_returns[scenario_type].append(normalized)
            self._scenario_counts[scenario_type] += 1
        self._episode_count += 1

        if self._episode_count % self.difficulty_update_interval == 0:
            self._update_difficulties()

    def _update_difficulties(self) -> None:
        for i in range(self.num_scenario_types):
            returns = list(self._episode_returns[i])
            if len(returns) < 5:
                continue
            win_rate = float(np.mean([r > 0 for r in returns]))
            if win_rate > 0.65:
                self._difficulties[i] = min(1.0, self._difficulties[i] + 0.05)
            elif win_rate < 0.35:
                self._difficulties[i] = max(0.0, self._difficulties[i] - 0.05)

    def get_difficulty(self, scenario_type: int) -> float:
        if 0 <= scenario_type < self.num_scenario_types:
            return float(self._difficulties[scenario_type])
        return 0.3

    def get_global_difficulty(self) -> float:
        return float(self._difficulties.mean())

    def sample_scenario(self) -> int:
        """Sample a scenario type weighted by difficulty."""
        diffs = self._difficulties
        weights = diffs / max(diffs.sum(), 1e-10)
        return int(np.random.choice(self.num_scenario_types, p=weights))

    def get_summary(self) -> Dict[str, Any]:
        return {
            "episode_count": self._episode_count,
            "global_difficulty": self.get_global_difficulty(),
            "per_scenario_difficulty": self._difficulties.tolist(),
            "per_scenario_episodes": self._scenario_counts.tolist(),
        }


# ---------------------------------------------------------------------------
# Extended: RegimeCalibrator
# ---------------------------------------------------------------------------

class RegimeCalibrator:
    """
    Calibrates regime parameters from historical market data.

    Takes a time series of returns/spreads and fits regime parameters
    to match the real data distribution, improving sim fidelity.
    """

    def __init__(
        self,
        num_regimes: int = 4,
        min_obs_per_regime: int = 50,
    ) -> None:
        self.num_regimes = num_regimes
        self.min_obs_per_regime = min_obs_per_regime
        self._fitted: bool = False
        self._regime_params: List[Dict[str, float]] = []

    def fit(
        self,
        returns: np.ndarray,
        spreads: np.ndarray,
        volumes: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Fit regime parameters from historical data using k-means clustering.

        Returns fitted parameters dict.
        """
        n = len(returns)
        if n < self.num_regimes * self.min_obs_per_regime:
            return {"error": f"insufficient_data: {n}"}

        # Features for regime classification
        features = np.column_stack([
            np.abs(returns),
            spreads,
            returns,
        ])

        # Normalize
        feat_mean = features.mean(axis=0)
        feat_std = features.std(axis=0) + 1e-8
        features_norm = (features - feat_mean) / feat_std

        # Simple k-means
        try:
            from sklearn.cluster import KMeans  # type: ignore[import]
            km = KMeans(n_clusters=self.num_regimes, random_state=42, n_init=10)
            labels = km.fit_predict(features_norm)
        except ImportError:
            # Fallback: quantile-based regime assignment
            vol_quantiles = np.quantile(np.abs(returns), np.linspace(0, 1, self.num_regimes + 1))
            labels = np.digitize(np.abs(returns), vol_quantiles[:-1]) - 1
            labels = np.clip(labels, 0, self.num_regimes - 1)

        # Fit parameters per regime
        self._regime_params = []
        for k in range(self.num_regimes):
            mask = labels == k
            if mask.sum() < self.min_obs_per_regime:
                self._regime_params.append({
                    "vol": float(np.std(returns)),
                    "drift": 0.0,
                    "spread_mult": 1.0,
                    "count": int(mask.sum()),
                })
                continue
            self._regime_params.append({
                "vol": float(np.std(returns[mask])),
                "drift": float(np.mean(returns[mask])),
                "spread_mult": float(np.mean(spreads[mask]) / max(np.mean(spreads), 1e-8)),
                "count": int(mask.sum()),
            })

        self._fitted = True
        return {
            "fitted": True,
            "regime_params": self._regime_params,
            "num_regimes": self.num_regimes,
        }

    def get_regime_config(self) -> RegimeConfig:
        """Convert fitted parameters to a RegimeConfig."""
        if not self._fitted or not self._regime_params:
            return RegimeConfig()

        # Map fitted regimes to MarketRegime enum
        # Sort by volatility
        sorted_params = sorted(self._regime_params, key=lambda p: p["vol"])
        n = len(sorted_params)

        calm_vol = sorted_params[0]["vol"] if n > 0 else 0.01
        volatile_vol = sorted_params[-1]["vol"] if n > 1 else 0.05

        cfg = RegimeConfig(
            calm_vol=calm_vol,
            volatile_vol=volatile_vol,
            crash_vol=volatile_vol * 1.5,
        )
        return cfg

    @property
    def is_fitted(self) -> bool:
        return self._fitted
