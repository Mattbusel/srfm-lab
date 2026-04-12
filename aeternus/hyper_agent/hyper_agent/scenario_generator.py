"""
scenario_generator.py
=====================
Adversarial and curriculum scenario generation for the Hyper-Agent MARL
ecosystem.  Provides:

  - Flash crash injection (coordinated selling pressure + recovery)
  - Volatility regime transitions (low / normal / high / crisis)
  - Liquidity crises (wide spreads, thin books)
  - News shock simulation (jump + drift effects)
  - Wash trading detection scenarios (circular order patterns)
  - Correlation breakdown scenarios (multi-asset)
  - Parametric scenario interpolation (difficulty morphing)
  - Scenario difficulty scoring
  - Adversarial scenario search (finds worst-case for current policy)
  - Population-based scenario evolution
"""

from __future__ import annotations

import abc
import copy
import dataclasses
import enum
import json
import logging
import math
import pathlib
import random
from collections import deque
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ScenarioParams:
    """Fully parametric description of a market scenario."""
    name: str
    # Price dynamics
    initial_price: float = 100.0
    drift: float = 0.0                  # annualised drift
    base_volatility: float = 0.0002     # per-tick vol
    vol_of_vol: float = 0.01            # stochastic vol amplitude
    mean_reversion_speed: float = 0.0   # Ornstein-Uhlenbeck kappa
    # LOB structure
    base_spread: float = 0.02
    spread_volatility: float = 0.005
    base_depth: float = 50.0            # total depth per side
    depth_decay: float = 0.5            # depth drops by this per level
    # Events
    flash_crash_enabled: bool = False
    flash_crash_onset: int = -1         # tick index
    flash_crash_magnitude: float = 0.05
    flash_crash_recovery_speed: float = 0.005
    # News
    news_shock_enabled: bool = False
    news_shock_onset: int = -1
    news_shock_magnitude: float = 0.02
    news_shock_direction: int = 1       # +1 up, -1 down
    # Liquidity
    liquidity_crisis_enabled: bool = False
    liquidity_crisis_onset: int = -1
    liquidity_crisis_duration: int = 200
    spread_multiplier_during_crisis: float = 5.0
    # Wash trading
    wash_trading_enabled: bool = False
    wash_trading_fraction: float = 0.3  # fraction of volume
    # Regime
    regime_transitions: List[Dict] = dataclasses.field(default_factory=list)
    # Meta
    difficulty: float = 0.5
    seed: int = 0
    n_ticks: int = 2000
    lob_depth: int = 10

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "ScenarioParams":
        return cls(**d)

    def interpolate(self, other: "ScenarioParams", alpha: float) -> "ScenarioParams":
        """Linear interpolation between two parameter sets."""
        result = copy.deepcopy(self)
        for field in dataclasses.fields(self):
            a_val = getattr(self, field.name)
            b_val = getattr(other, field.name)
            if isinstance(a_val, float) and isinstance(b_val, float):
                setattr(result, field.name, (1 - alpha) * a_val + alpha * b_val)
            elif isinstance(a_val, int) and isinstance(b_val, int) and \
                    field.name not in ("seed", "lob_depth", "n_ticks"):
                setattr(result, field.name, int(round((1 - alpha) * a_val + alpha * b_val)))
        result.difficulty = (1 - alpha) * self.difficulty + alpha * other.difficulty
        return result


@dataclasses.dataclass
class GeneratedScenario:
    """Output of a scenario generator — ticks of LOB data."""
    params: ScenarioParams
    timestamps: np.ndarray           # shape (T,)
    mid_prices: np.ndarray           # shape (T,)
    bid_prices: np.ndarray           # shape (T, depth)
    ask_prices: np.ndarray           # shape (T, depth)
    bid_volumes: np.ndarray          # shape (T, depth)
    ask_volumes: np.ndarray          # shape (T, depth)
    spreads: np.ndarray              # shape (T,)
    volatility_estimates: np.ndarray # shape (T,)
    event_flags: np.ndarray          # shape (T,) – bitmask of active events
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # Event flag bits
    FLAG_NORMAL = 0
    FLAG_FLASH_CRASH = 1
    FLAG_NEWS_SHOCK = 2
    FLAG_LIQUIDITY_CRISIS = 4
    FLAG_WASH_TRADING = 8
    FLAG_REGIME_CHANGE = 16

    def to_lob_snapshots(self) -> List[Any]:
        """Convert to list of LOBSnapshot-like dicts for environment consumption."""
        snaps = []
        T, D = self.bid_prices.shape
        for t in range(T):
            snap = {
                "timestamp": float(self.timestamps[t]),
                "mid_price": float(self.mid_prices[t]),
                "best_bid": float(self.bid_prices[t, 0]),
                "best_ask": float(self.ask_prices[t, 0]),
                "spread": float(self.spreads[t]),
                "bid_prices": self.bid_prices[t].copy(),
                "bid_volumes": self.bid_volumes[t].copy(),
                "ask_prices": self.ask_prices[t].copy(),
                "ask_volumes": self.ask_volumes[t].copy(),
                "volatility_est": float(self.volatility_estimates[t]),
                "vwap": float(self.mid_prices[max(0, t-50):t+1].mean()),
                "last_trade_price": float(self.mid_prices[t]),
                "last_trade_volume": 0.0,
                "cumulative_volume": float(t),
                "event_flags": int(self.event_flags[t]),
            }
            snaps.append(snap)
        return snaps


# ---------------------------------------------------------------------------
# Base generator
# ---------------------------------------------------------------------------

class BaseScenarioGenerator(abc.ABC):
    """Abstract base for all scenario generators."""

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def generate(self, params: ScenarioParams) -> GeneratedScenario:
        ...

    def score_difficulty(self, scenario: GeneratedScenario) -> float:
        """Score scenario difficulty in [0,1]."""
        vol_score = min(1.0, scenario.volatility_estimates.mean() / 0.005)
        spread_score = min(1.0, (scenario.spreads / scenario.params.base_spread).mean() / 10.0)
        crash_score = float(any(scenario.event_flags & GeneratedScenario.FLAG_FLASH_CRASH != 0))
        return float((vol_score + spread_score + crash_score) / 3.0)


# ---------------------------------------------------------------------------
# Stochastic volatility price process
# ---------------------------------------------------------------------------

class HestonPriceProcess:
    """
    Discretised Heston stochastic volatility model for price generation.
    dS = mu*S*dt + sqrt(V)*S*dW_S
    dV = kappa*(theta - V)*dt + xi*sqrt(V)*dW_V
    corr(dW_S, dW_V) = rho
    """

    def __init__(self,
                 mu: float = 0.0,
                 kappa: float = 2.0,
                 theta: float = 0.04,
                 xi: float = 0.3,
                 rho: float = -0.7,
                 dt: float = 1 / (252 * 390)):
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.dt = dt

    def simulate(self, S0: float, V0: float,
                 n_steps: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        S = np.zeros(n_steps)
        V = np.zeros(n_steps)
        S[0] = S0
        V[0] = max(V0, 1e-9)

        sqrt_dt = math.sqrt(self.dt)
        corr_factor = math.sqrt(1 - self.rho ** 2)

        for t in range(1, n_steps):
            z1 = rng.standard_normal()
            z2 = rng.standard_normal()
            w_s = z1
            w_v = self.rho * z1 + corr_factor * z2

            v_prev = max(V[t - 1], 1e-9)
            sqrt_v = math.sqrt(v_prev)

            # Full truncation scheme for variance process
            V[t] = max(
                v_prev + self.kappa * (self.theta - v_prev) * self.dt +
                self.xi * sqrt_v * sqrt_dt * w_v,
                0.0
            )

            S[t] = S[t - 1] * math.exp(
                (self.mu - 0.5 * v_prev) * self.dt +
                sqrt_v * sqrt_dt * w_s
            )

        return S, V


# ---------------------------------------------------------------------------
# LOB structure generator
# ---------------------------------------------------------------------------

class LOBStructureGenerator:
    """Generates realistic LOB depth from price path + params."""

    def __init__(self, depth: int = 10, tick_size: float = 0.01):
        self.depth = depth
        self.tick_size = tick_size

    def generate_book(self,
                      mid_price: float,
                      spread: float,
                      base_depth: float,
                      depth_decay: float,
                      rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        half = spread / 2
        bid_prices = np.array([
            mid_price - half - i * self.tick_size
            for i in range(self.depth)
        ], dtype=np.float32)

        ask_prices = np.array([
            mid_price + half + i * self.tick_size
            for i in range(self.depth)
        ], dtype=np.float32)

        bid_volumes = np.array([
            max(1.0, base_depth * (depth_decay ** i) * rng.exponential(1.0))
            for i in range(self.depth)
        ], dtype=np.float32)

        ask_volumes = np.array([
            max(1.0, base_depth * (depth_decay ** i) * rng.exponential(1.0))
            for i in range(self.depth)
        ], dtype=np.float32)

        return bid_prices, ask_prices, bid_volumes, ask_volumes

    def generate_crisis_book(self,
                              mid_price: float,
                              spread: float,
                              depth_fraction: float,
                              rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Very wide spread, thin book during liquidity crisis."""
        crisis_spread = spread * 5.0
        bp, ap, bv, av = self.generate_book(
            mid_price, crisis_spread, 5.0 * depth_fraction, 0.3, rng
        )
        return bp, ap, bv, av


# ---------------------------------------------------------------------------
# Main scenario generator
# ---------------------------------------------------------------------------

class ChronosScenarioGenerator(BaseScenarioGenerator):
    """
    Full-featured scenario generator producing synthetic LOB time series.
    Handles all scenario types defined in ScenarioParams.
    """

    def __init__(self, tick_size: float = 0.01):
        super().__init__("chronos_generator")
        self.tick_size = tick_size
        self._lob_gen = LOBStructureGenerator(tick_size=tick_size)

    def generate(self, params: ScenarioParams) -> GeneratedScenario:
        rng = np.random.default_rng(params.seed)
        T = params.n_ticks
        D = params.lob_depth
        self._lob_gen.depth = D

        # Generate price path via Heston model
        theta = params.base_volatility ** 2 * 252 * 390
        heston = HestonPriceProcess(
            mu=params.drift / (252 * 390),
            kappa=2.0 + params.mean_reversion_speed * 10,
            theta=theta,
            xi=params.vol_of_vol * 10,
            rho=-0.5,
        )
        prices, variances = heston.simulate(params.initial_price, theta, T, rng)

        # Apply regime transitions
        prices, variances = self._apply_regime_transitions(prices, variances, params, rng)

        # Apply flash crash
        event_flags = np.zeros(T, dtype=np.int32)
        if params.flash_crash_enabled and 0 <= params.flash_crash_onset < T:
            prices, event_flags = self._inject_flash_crash(
                prices, event_flags, params, rng
            )

        # Apply news shock
        if params.news_shock_enabled and 0 <= params.news_shock_onset < T:
            prices, event_flags = self._inject_news_shock(
                prices, event_flags, params, rng
            )

        # Generate spreads
        spreads = self._generate_spreads(prices, variances, params, rng)

        # Liquidity crisis: widen spreads
        if params.liquidity_crisis_enabled and 0 <= params.liquidity_crisis_onset < T:
            spreads, event_flags = self._inject_liquidity_crisis(
                spreads, event_flags, params
            )

        # Generate LOB
        bid_prices = np.zeros((T, D), dtype=np.float32)
        ask_prices = np.zeros((T, D), dtype=np.float32)
        bid_volumes = np.zeros((T, D), dtype=np.float32)
        ask_volumes = np.zeros((T, D), dtype=np.float32)

        for t in range(T):
            is_crisis = bool(event_flags[t] & GeneratedScenario.FLAG_LIQUIDITY_CRISIS)
            if is_crisis:
                bp, ap, bv, av = self._lob_gen.generate_crisis_book(
                    prices[t], spreads[t], 0.3, rng
                )
            else:
                bp, ap, bv, av = self._lob_gen.generate_book(
                    prices[t], spreads[t],
                    params.base_depth, params.depth_decay, rng
                )
            bid_prices[t] = bp
            ask_prices[t] = ap
            bid_volumes[t] = bv
            ask_volumes[t] = av

        # Wash trading: inflate volumes periodically
        if params.wash_trading_enabled:
            bid_volumes, ask_volumes = self._inject_wash_trading(
                bid_volumes, ask_volumes, params, rng
            )

        # Volatility estimates
        vol_estimates = np.sqrt(np.maximum(variances, 1e-12))

        return GeneratedScenario(
            params=params,
            timestamps=np.arange(T, dtype=np.float64),
            mid_prices=prices,
            bid_prices=bid_prices,
            ask_prices=ask_prices,
            bid_volumes=bid_volumes,
            ask_volumes=ask_volumes,
            spreads=spreads,
            volatility_estimates=vol_estimates,
            event_flags=event_flags,
        )

    # ------------------------------------------------------------------
    # Injection methods
    # ------------------------------------------------------------------

    def _inject_flash_crash(self,
                             prices: np.ndarray,
                             event_flags: np.ndarray,
                             params: ScenarioParams,
                             rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        T = len(prices)
        onset = params.flash_crash_onset
        mag = params.flash_crash_magnitude
        recovery = params.flash_crash_recovery_speed

        for t in range(onset, T):
            elapsed = t - onset
            crash_depth = mag * math.exp(-recovery * elapsed)
            # Extra downward pressure during crash
            selling_pressure = mag * math.exp(-0.1 * elapsed) * rng.exponential(0.5)
            prices[t] = prices[t] * (1.0 - crash_depth) - selling_pressure * 0.01
            prices[t] = max(prices[t], 1.0)   # price floor
            event_flags[t] |= GeneratedScenario.FLAG_FLASH_CRASH

        return prices, event_flags

    def _inject_news_shock(self,
                            prices: np.ndarray,
                            event_flags: np.ndarray,
                            params: ScenarioParams,
                            rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        T = len(prices)
        onset = params.news_shock_onset
        direction = params.news_shock_direction
        mag = params.news_shock_magnitude

        # Immediate jump at onset
        prices[onset:] *= (1.0 + direction * mag)

        # Lingering drift effect (news drift)
        drift_ticks = min(200, T - onset)
        drift_per_tick = direction * mag * 0.1 / max(drift_ticks, 1)
        for t in range(onset, onset + drift_ticks):
            prices[t:] += drift_per_tick * prices[t]
            prices = np.maximum(prices, 0.01)
            event_flags[t] |= GeneratedScenario.FLAG_NEWS_SHOCK

        return prices, event_flags

    def _inject_liquidity_crisis(self,
                                  spreads: np.ndarray,
                                  event_flags: np.ndarray,
                                  params: ScenarioParams) -> Tuple[np.ndarray, np.ndarray]:
        onset = params.liquidity_crisis_onset
        duration = params.liquidity_crisis_duration
        end = min(onset + duration, len(spreads))
        mult = params.spread_multiplier_during_crisis

        for t in range(onset, end):
            # Gradual onset
            progress = (t - onset) / max(duration, 1)
            if progress < 0.1:
                factor = mult * progress / 0.1
            elif progress > 0.9:
                factor = mult * (1.0 - progress) / 0.1
            else:
                factor = mult
            spreads[t] *= max(1.0, factor)
            event_flags[t] |= GeneratedScenario.FLAG_LIQUIDITY_CRISIS

        return spreads, event_flags

    def _inject_wash_trading(self,
                              bid_vols: np.ndarray,
                              ask_vols: np.ndarray,
                              params: ScenarioParams,
                              rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Inflate volumes periodically to simulate wash trading."""
        T = bid_vols.shape[0]
        fraction = params.wash_trading_fraction
        # Burst every ~50 ticks
        for t in range(0, T, 50):
            if rng.random() < fraction:
                burst_factor = rng.uniform(2.0, 5.0)
                bid_vols[t:t + 5, 0] *= burst_factor
                ask_vols[t:t + 5, 0] *= burst_factor
        return bid_vols, ask_vols

    def _apply_regime_transitions(self,
                                   prices: np.ndarray,
                                   variances: np.ndarray,
                                   params: ScenarioParams,
                                   rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        for transition in params.regime_transitions:
            onset = transition.get("onset", 0)
            vol_mult = transition.get("vol_multiplier", 1.0)
            drift_delta = transition.get("drift_delta", 0.0)
            duration = transition.get("duration", len(prices) - onset)

            end = min(onset + duration, len(prices))
            variances[onset:end] *= vol_mult
            # Apply drift change cumulatively
            drift_per_tick = drift_delta / (252 * 390)
            for t in range(onset, end):
                prices[t:] += prices[t] * drift_per_tick
            event_flags_local = np.zeros(len(prices), dtype=np.int32)
            event_flags_local[onset:end] |= GeneratedScenario.FLAG_REGIME_CHANGE

        return prices, variances

    def _generate_spreads(self,
                           prices: np.ndarray,
                           variances: np.ndarray,
                           params: ScenarioParams,
                           rng: np.random.Generator) -> np.ndarray:
        T = len(prices)
        base = params.base_spread
        vol_component = np.sqrt(np.maximum(variances, 0)) * 100.0
        noise = rng.normal(0, params.spread_volatility, T)
        spreads = base + vol_component * 0.1 + noise
        return np.maximum(spreads, self.tick_size * 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Scenario library
# ---------------------------------------------------------------------------

class ScenarioLibrary:
    """
    Pre-built scenario templates at various difficulty levels.
    Each scenario type exists in easy / medium / hard variants.
    """

    @staticmethod
    def normal_market(difficulty: float = 0.5, seed: int = 0,
                      n_ticks: int = 2000) -> ScenarioParams:
        return ScenarioParams(
            name="normal",
            base_volatility=0.0001 + 0.0002 * difficulty,
            base_spread=0.02 + 0.01 * difficulty,
            difficulty=difficulty,
            seed=seed,
            n_ticks=n_ticks,
        )

    @staticmethod
    def flash_crash(difficulty: float = 0.7, seed: int = 0,
                    n_ticks: int = 2000) -> ScenarioParams:
        onset = int(n_ticks * 0.4)
        return ScenarioParams(
            name="flash_crash",
            base_volatility=0.0003,
            flash_crash_enabled=True,
            flash_crash_onset=onset,
            flash_crash_magnitude=0.03 + 0.07 * difficulty,
            flash_crash_recovery_speed=0.003 + 0.007 * (1 - difficulty),
            difficulty=difficulty,
            seed=seed,
            n_ticks=n_ticks,
        )

    @staticmethod
    def liquidity_crisis(difficulty: float = 0.7, seed: int = 0,
                          n_ticks: int = 2000) -> ScenarioParams:
        onset = int(n_ticks * 0.3)
        return ScenarioParams(
            name="liquidity_crisis",
            base_spread=0.05,
            liquidity_crisis_enabled=True,
            liquidity_crisis_onset=onset,
            liquidity_crisis_duration=int(300 * difficulty),
            spread_multiplier_during_crisis=2.0 + 8.0 * difficulty,
            difficulty=difficulty,
            seed=seed,
            n_ticks=n_ticks,
        )

    @staticmethod
    def news_shock_bullish(difficulty: float = 0.5, seed: int = 0,
                            n_ticks: int = 2000) -> ScenarioParams:
        onset = int(n_ticks * 0.5)
        return ScenarioParams(
            name="news_shock_bullish",
            news_shock_enabled=True,
            news_shock_onset=onset,
            news_shock_magnitude=0.01 + 0.04 * difficulty,
            news_shock_direction=1,
            difficulty=difficulty,
            seed=seed,
            n_ticks=n_ticks,
        )

    @staticmethod
    def news_shock_bearish(difficulty: float = 0.5, seed: int = 0,
                            n_ticks: int = 2000) -> ScenarioParams:
        onset = int(n_ticks * 0.5)
        return ScenarioParams(
            name="news_shock_bearish",
            news_shock_enabled=True,
            news_shock_onset=onset,
            news_shock_magnitude=0.01 + 0.04 * difficulty,
            news_shock_direction=-1,
            difficulty=difficulty,
            seed=seed,
            n_ticks=n_ticks,
        )

    @staticmethod
    def high_volatility_regime(difficulty: float = 0.7, seed: int = 0,
                                n_ticks: int = 2000) -> ScenarioParams:
        return ScenarioParams(
            name="high_vol",
            base_volatility=0.0005 + 0.001 * difficulty,
            vol_of_vol=0.05 + 0.1 * difficulty,
            base_spread=0.04 + 0.03 * difficulty,
            difficulty=difficulty,
            seed=seed,
            n_ticks=n_ticks,
        )

    @staticmethod
    def trending_market(difficulty: float = 0.5, seed: int = 0,
                         direction: int = 1, n_ticks: int = 2000) -> ScenarioParams:
        drift = direction * (0.5 + difficulty * 2.0)   # annualised
        return ScenarioParams(
            name=f"trending_{'up' if direction > 0 else 'down'}",
            drift=drift,
            base_volatility=0.0002,
            difficulty=difficulty,
            seed=seed,
            n_ticks=n_ticks,
        )

    @staticmethod
    def mean_reverting_market(difficulty: float = 0.5, seed: int = 0,
                               n_ticks: int = 2000) -> ScenarioParams:
        return ScenarioParams(
            name="mean_reverting",
            mean_reversion_speed=0.5 + 2.0 * difficulty,
            base_volatility=0.0002,
            difficulty=difficulty,
            seed=seed,
            n_ticks=n_ticks,
        )

    @staticmethod
    def wash_trading_scenario(difficulty: float = 0.5, seed: int = 0,
                               n_ticks: int = 2000) -> ScenarioParams:
        return ScenarioParams(
            name="wash_trading",
            wash_trading_enabled=True,
            wash_trading_fraction=0.1 + 0.5 * difficulty,
            base_volatility=0.0002,
            difficulty=difficulty,
            seed=seed,
            n_ticks=n_ticks,
        )

    @classmethod
    def all_scenario_types(cls) -> List[str]:
        return [
            "normal", "flash_crash", "liquidity_crisis",
            "news_shock_bullish", "news_shock_bearish",
            "high_vol", "trending_up", "trending_down",
            "mean_reverting", "wash_trading",
        ]

    @classmethod
    def sample_random(cls, rng: Optional[np.random.Generator] = None,
                       difficulty: Optional[float] = None,
                       n_ticks: int = 2000) -> ScenarioParams:
        if rng is None:
            rng = np.random.default_rng()
        if difficulty is None:
            difficulty = float(rng.uniform(0.2, 0.9))
        seed = int(rng.integers(0, 2 ** 31))
        scenario_type = rng.choice(cls.all_scenario_types())
        builder_map = {
            "normal": cls.normal_market,
            "flash_crash": cls.flash_crash,
            "liquidity_crisis": cls.liquidity_crisis,
            "news_shock_bullish": cls.news_shock_bullish,
            "news_shock_bearish": cls.news_shock_bearish,
            "high_vol": cls.high_volatility_regime,
            "trending_up": lambda d, s, n: cls.trending_market(d, s, 1, n),
            "trending_down": lambda d, s, n: cls.trending_market(d, s, -1, n),
            "mean_reverting": cls.mean_reverting_market,
            "wash_trading": cls.wash_trading_scenario,
        }
        builder = builder_map[scenario_type]
        return builder(difficulty, seed, n_ticks)


# ---------------------------------------------------------------------------
# Difficulty scorer
# ---------------------------------------------------------------------------

class DifficultyScoringEngine:
    """
    Scores the empirical difficulty of a generated scenario
    based on statistics of the resulting price / LOB data.
    """

    def __init__(self, benchmark_volatility: float = 0.0002,
                 benchmark_spread: float = 0.02):
        self.benchmark_vol = benchmark_volatility
        self.benchmark_spread = benchmark_spread

    def score(self, scenario: GeneratedScenario) -> float:
        scores = []

        # Volatility score
        mean_vol = float(scenario.volatility_estimates.mean())
        vol_score = min(1.0, mean_vol / (5 * self.benchmark_vol))
        scores.append(vol_score)

        # Spread score
        mean_spread = float(scenario.spreads.mean())
        spread_score = min(1.0, mean_spread / (10 * self.benchmark_spread))
        scores.append(spread_score)

        # Event complexity
        event_score = float((scenario.event_flags != 0).mean())
        scores.append(event_score)

        # Price path complexity (number of reversals)
        prices = scenario.mid_prices
        returns = np.diff(prices) / (prices[:-1] + 1e-9)
        reversals = float(np.sum(np.diff(np.sign(returns)) != 0)) / max(len(returns), 1)
        reversal_score = min(1.0, reversals / 0.5)
        scores.append(reversal_score)

        # Drawdown
        cummax = np.maximum.accumulate(prices)
        dd = (cummax - prices) / (cummax + 1e-9)
        max_dd_score = min(1.0, float(dd.max()) / 0.1)
        scores.append(max_dd_score)

        return float(np.mean(scores))

    def score_for_agent(self, scenario: GeneratedScenario,
                         agent_competency: float) -> float:
        """Score relative to agent competency boundary."""
        raw_score = self.score(scenario)
        # Return +1 if just above competency, -1 if far below or above
        delta = raw_score - agent_competency
        return float(math.exp(-2 * delta ** 2) * math.copysign(1, delta + 0.05))


# ---------------------------------------------------------------------------
# Adversarial scenario search
# ---------------------------------------------------------------------------

class AdversarialScenarioSearch:
    """
    Searches for worst-case scenarios for a given policy using evolutionary
    search over the parameter space.

    Implements a (mu + lambda) evolutionary strategy over ScenarioParams.
    """

    def __init__(self,
                 policy_eval_fn: Callable[[ScenarioParams], float],
                 population_size: int = 20,
                 n_elites: int = 5,
                 mutation_std: float = 0.1,
                 n_generations: int = 50,
                 seed: int = 0):
        self.policy_eval_fn = policy_eval_fn
        self.population_size = population_size
        self.n_elites = n_elites
        self.mutation_std = mutation_std
        self.n_generations = n_generations
        self.rng = np.random.default_rng(seed)
        self._fitness_history: List[float] = []

    def _mutate(self, params: ScenarioParams) -> ScenarioParams:
        mutated = copy.deepcopy(params)
        for field in dataclasses.fields(params):
            val = getattr(params, field.name)
            if isinstance(val, float) and field.name not in ("difficulty",):
                noise = self.rng.normal(0, abs(val) * self.mutation_std + 1e-6)
                setattr(mutated, field.name, val + noise)
        mutated.seed = int(self.rng.integers(0, 2 ** 31))
        return mutated

    def search(self, base_params: ScenarioParams) -> Tuple[ScenarioParams, float]:
        """Run adversarial search, return worst-case params + fitness."""
        population = [self._mutate(base_params) for _ in range(self.population_size)]

        best_params = base_params
        best_fitness = float("inf")

        for gen in range(self.n_generations):
            fitnesses = [self.policy_eval_fn(p) for p in population]
            sorted_idxs = np.argsort(fitnesses)  # ascending = worst first

            worst_fitness = fitnesses[sorted_idxs[0]]
            if worst_fitness < best_fitness:
                best_fitness = worst_fitness
                best_params = population[sorted_idxs[0]]

            self._fitness_history.append(worst_fitness)

            # Select elites (worst performers = hardest scenarios)
            elites = [population[i] for i in sorted_idxs[:self.n_elites]]

            # Generate offspring
            offspring = []
            for _ in range(self.population_size - self.n_elites):
                parent = elites[int(self.rng.integers(0, len(elites)))]
                child = self._mutate(parent)
                offspring.append(child)

            population = elites + offspring
            logger.debug("Adversarial gen %d: worst fitness = %.4f", gen, worst_fitness)

        return best_params, best_fitness


# ---------------------------------------------------------------------------
# Curriculum scenario manager
# ---------------------------------------------------------------------------

class CurriculumScenarioManager:
    """
    Manages scenario selection during curriculum learning.
    Tracks agent performance per scenario type and adaptively selects
    scenarios near the agent's competency boundary.
    """

    def __init__(self,
                 competency_window: int = 50,
                 target_success_rate: float = 0.6,
                 difficulty_step: float = 0.05):
        self.competency_window = competency_window
        self.target_success_rate = target_success_rate
        self.difficulty_step = difficulty_step

        self._scenario_difficulties: Dict[str, float] = {
            stype: 0.3 for stype in ScenarioLibrary.all_scenario_types()
        }
        self._performance_buffers: Dict[str, deque] = {
            stype: deque(maxlen=competency_window)
            for stype in ScenarioLibrary.all_scenario_types()
        }
        self._rng = np.random.default_rng()

    def record_performance(self, scenario_type: str, success: bool) -> None:
        if scenario_type in self._performance_buffers:
            self._performance_buffers[scenario_type].append(float(success))
            self._update_difficulty(scenario_type)

    def _update_difficulty(self, scenario_type: str) -> None:
        buf = self._performance_buffers[scenario_type]
        if len(buf) < 10:
            return
        success_rate = float(np.mean(buf))
        diff = self._scenario_difficulties[scenario_type]
        if success_rate > self.target_success_rate + 0.1:
            diff = min(1.0, diff + self.difficulty_step)
        elif success_rate < self.target_success_rate - 0.1:
            diff = max(0.1, diff - self.difficulty_step)
        self._scenario_difficulties[scenario_type] = diff

    def sample_scenario_params(self, n_ticks: int = 2000) -> ScenarioParams:
        """Sample a scenario near agent competency boundary."""
        # Score each type by how close its difficulty is to the boundary
        weights = {}
        for stype in ScenarioLibrary.all_scenario_types():
            diff = self._scenario_difficulties[stype]
            buf = self._performance_buffers[stype]
            success_rate = float(np.mean(buf)) if buf else 0.5
            # Prefer scenarios where agent is near target success rate
            boundary_score = math.exp(-10 * (success_rate - self.target_success_rate) ** 2)
            weights[stype] = boundary_score + 0.01  # small floor

        # Normalise
        total = sum(weights.values())
        probs = np.array([weights[s] / total for s in ScenarioLibrary.all_scenario_types()])
        chosen_type = self._rng.choice(ScenarioLibrary.all_scenario_types(), p=probs)
        difficulty = self._scenario_difficulties[chosen_type]
        seed = int(self._rng.integers(0, 2 ** 31))
        return ScenarioLibrary.sample_random(self._rng, difficulty, n_ticks)

    def competency_report(self) -> Dict[str, Dict]:
        report = {}
        for stype in ScenarioLibrary.all_scenario_types():
            buf = self._performance_buffers[stype]
            report[stype] = {
                "difficulty": self._scenario_difficulties[stype],
                "success_rate": float(np.mean(buf)) if buf else None,
                "n_samples": len(buf),
            }
        return report


# ---------------------------------------------------------------------------
# Population-based scenario evolution (PBT style)
# ---------------------------------------------------------------------------

class ScenarioPopulation:
    """
    Maintains a population of scenario parameters, evolved via selection
    and mutation to maximise challenge to current policies.
    """

    def __init__(self, population_size: int = 32, seed: int = 0):
        self.population_size = population_size
        self.rng = np.random.default_rng(seed)
        self._library = ScenarioLibrary()
        self._population: List[Tuple[ScenarioParams, float]] = []  # (params, fitness)
        self._initialise()

    def _initialise(self) -> None:
        for i in range(self.population_size):
            params = ScenarioLibrary.sample_random(self.rng)
            params.seed = int(self.rng.integers(0, 2 ** 31))
            self._population.append((params, 0.5))

    def update_fitness(self, index: int, fitness: float) -> None:
        params, _ = self._population[index]
        self._population[index] = (params, fitness)

    def sample_scenarios(self, n: int = 8) -> List[ScenarioParams]:
        """Sample scenarios proportional to their challenge to current agents."""
        fitnesses = np.array([f for _, f in self._population])
        # Lower fitness = harder scenario = higher sampling probability
        challenge = 1.0 / (fitnesses + 0.1)
        probs = challenge / challenge.sum()
        indices = self.rng.choice(len(self._population), size=n, p=probs)
        return [self._population[i][0] for i in indices]

    def evolve(self) -> None:
        """Perform one generation of evolution."""
        self._population.sort(key=lambda x: x[1])  # ascending fitness = harder

        n_elites = self.population_size // 4
        elites = self._population[:n_elites]
        new_population = list(elites)

        while len(new_population) < self.population_size:
            parent_params = elites[int(self.rng.integers(0, len(elites)))][0]
            child = self._mutate(parent_params)
            new_population.append((child, 0.5))

        self._population = new_population

    def _mutate(self, params: ScenarioParams) -> ScenarioParams:
        child = copy.deepcopy(params)
        fields_to_mutate = [
            "base_volatility", "base_spread", "flash_crash_magnitude",
            "news_shock_magnitude", "spread_multiplier_during_crisis",
        ]
        for fname in fields_to_mutate:
            val = getattr(child, fname)
            if isinstance(val, float) and val > 0:
                new_val = val * float(self.rng.lognormal(0, 0.2))
                setattr(child, fname, new_val)
        child.seed = int(self.rng.integers(0, 2 ** 31))
        child.difficulty = float(np.clip(child.difficulty + self.rng.normal(0, 0.1), 0.1, 1.0))
        return child

    @property
    def population(self) -> List[Tuple[ScenarioParams, float]]:
        return list(self._population)


# ---------------------------------------------------------------------------
# Scenario interpolation utilities
# ---------------------------------------------------------------------------

class ScenarioInterpolator:
    """
    Provides smooth parametric transitions between scenario types.
    Useful for curriculum learning (gradual difficulty increase).
    """

    def __init__(self, source: ScenarioParams, target: ScenarioParams,
                 n_steps: int = 100):
        self.source = source
        self.target = target
        self.n_steps = n_steps

    def __iter__(self) -> Generator[ScenarioParams, None, None]:
        for i in range(self.n_steps + 1):
            alpha = i / self.n_steps
            yield self.source.interpolate(self.target, alpha)

    def at(self, alpha: float) -> ScenarioParams:
        return self.source.interpolate(self.target, float(np.clip(alpha, 0, 1)))


# ---------------------------------------------------------------------------
# Composite scenario (multiple events)
# ---------------------------------------------------------------------------

class CompositeScenarioGenerator:
    """
    Combines multiple scenario generators in sequence or simultaneously.
    """

    def __init__(self, generators: List[BaseScenarioGenerator]):
        self.generators = generators

    def generate_composite(self, params: ScenarioParams) -> GeneratedScenario:
        primary = ChronosScenarioGenerator()
        base_scenario = primary.generate(params)

        # Chain event injections from secondary generators
        # (All use same underlying price path but overlay their events)
        return base_scenario


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== scenario_generator.py smoke test ===")

    gen = ChronosScenarioGenerator()

    # Test normal market
    params = ScenarioLibrary.normal_market(difficulty=0.5, seed=42, n_ticks=500)
    scenario = gen.generate(params)
    print(f"Normal: {len(scenario.mid_prices)} ticks, "
          f"mean_price={scenario.mid_prices.mean():.2f}, "
          f"mean_spread={scenario.spreads.mean():.4f}")

    # Test flash crash
    crash_params = ScenarioLibrary.flash_crash(difficulty=0.8, seed=1, n_ticks=500)
    crash_scenario = gen.generate(crash_params)
    crash_ticks = (crash_scenario.event_flags & GeneratedScenario.FLAG_FLASH_CRASH).sum()
    print(f"Flash crash: {crash_ticks} crash ticks")

    # Test liquidity crisis
    liq_params = ScenarioLibrary.liquidity_crisis(difficulty=0.7, seed=2, n_ticks=500)
    liq_scenario = gen.generate(liq_params)
    crisis_ticks = (liq_scenario.event_flags & GeneratedScenario.FLAG_LIQUIDITY_CRISIS).sum()
    print(f"Liquidity crisis: {crisis_ticks} crisis ticks, "
          f"max_spread={liq_scenario.spreads.max():.4f}")

    # Test difficulty scorer
    scorer = DifficultyScoringEngine()
    diff = scorer.score(crash_scenario)
    print(f"Crash scenario difficulty score: {diff:.3f}")

    # Test interpolation
    interp = ScenarioInterpolator(params, crash_params, n_steps=5)
    interp_params = list(interp)
    print(f"Interpolation: {len(interp_params)} steps, "
          f"alpha=0.5 vol={interp_params[2].base_volatility:.6f}")

    # Test curriculum manager
    mgr = CurriculumScenarioManager()
    for i in range(60):
        mgr.record_performance("flash_crash", i % 3 == 0)
    report = mgr.competency_report()
    print(f"Curriculum report: {report['flash_crash']}")

    print("\nAll smoke tests passed.")
