"""
strategy_adapter.py -- LARSA strategy adapter for event-driven backtesting.

Bridges the LARSA signal logic (BH mass physics, CF cross, GARCH vol,
quaternion navigation) to the backtest engine's event system.
All computations use only past data to prevent lookahead bias.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .engine import (
    Direction,
    EventType,
    MarketEvent,
    SignalEvent,
)
from .data_handler import BarBuffer, MultiAssetData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LARSA regime and signal enumerations
# ---------------------------------------------------------------------------

class BHRegime(Enum):
    BH_ACTIVE = "BH_ACTIVE"       # Black Hole mass above threshold -- strong trend
    BH_INACTIVE = "BH_INACTIVE"   # Below threshold -- no clear trend
    TRANSITIONING = "TRANSITIONING"


class HurstRegime(Enum):
    TRENDING = "TRENDING"          # H > 0.6
    RANDOM_WALK = "RANDOM_WALK"    # 0.4 <= H <= 0.6
    MEAN_REVERTING = "MEAN_REVERTING"  # H < 0.4


# ---------------------------------------------------------------------------
# BH Mass Adapter -- Minkowski metric accumulation on historical bars
# ---------------------------------------------------------------------------

class BHMassAdapter:
    """
    Re-implements the Black Hole mass physics on historical bar data.

    The BH mass M represents accumulated momentum in a pseudo-Minkowski
    spacetime where the "time" axis is bar index and "space" axes are
    price, volume, and volatility.

    Mass update rule (discrete):
        M_t = decay * M_{t-1} + ds^2
        ds^2 = w_p*(dp/p)^2 + w_v*(dV/V)^2 + w_s*(dsigma/sigma)^2

    where dp = close_t - close_{t-1}, dV = volume difference,
    dsigma = realized vol difference.
    """

    def __init__(
        self,
        decay: float = 0.94,
        w_price: float = 1.0,
        w_volume: float = 0.3,
        w_vol: float = 0.5,
        mass_threshold: float = 0.0025,
        vol_window: int = 20,
    ):
        self.decay = decay
        self.w_price = w_price
        self.w_volume = w_volume
        self.w_vol = w_vol
        self.mass_threshold = mass_threshold
        self.vol_window = vol_window
        self._mass: Dict[str, float] = {}
        self._prev_sigma: Dict[str, float] = {}

    def compute_mass(self, symbol: str, buffer: BarBuffer) -> float:
        """
        Compute current BH mass for a symbol from its BarBuffer.
        Uses the last bar delta vs. the one before it.
        """
        if buffer.size < self.vol_window + 1:
            return 0.0

        closes = buffer.closes()
        volumes = buffer.volumes()

        # Realized volatility
        rets = np.diff(np.log(closes[-self.vol_window:] + 1e-12))
        sigma = float(np.std(rets)) if len(rets) > 1 else 1e-8

        # Current bar delta
        dp_rel = (closes[-1] - closes[-2]) / (closes[-2] + 1e-12)
        dv_rel = (volumes[-1] - volumes[-2]) / (volumes[-2] + 1e-8) if volumes[-2] > 0 else 0.0

        prev_sigma = self._prev_sigma.get(symbol, sigma)
        ds_sigma = (sigma - prev_sigma) / (prev_sigma + 1e-12)
        self._prev_sigma[symbol] = sigma

        # Minkowski interval squared (spacelike)
        ds2 = (
            self.w_price * dp_rel**2
            + self.w_volume * dv_rel**2
            + self.w_vol * ds_sigma**2
        )

        prev_mass = self._mass.get(symbol, 0.0)
        new_mass = self.decay * prev_mass + ds2
        self._mass[symbol] = new_mass
        return new_mass

    def get_regime(self, symbol: str) -> BHRegime:
        mass = self._mass.get(symbol, 0.0)
        if mass > self.mass_threshold * 1.2:
            return BHRegime.BH_ACTIVE
        elif mass > self.mass_threshold * 0.8:
            return BHRegime.TRANSITIONING
        else:
            return BHRegime.BH_INACTIVE

    def reset(self, symbol: Optional[str] = None) -> None:
        if symbol:
            self._mass.pop(symbol, None)
            self._prev_sigma.pop(symbol, None)
        else:
            self._mass.clear()
            self._prev_sigma.clear()


# ---------------------------------------------------------------------------
# Hurst Exponent estimator
# ---------------------------------------------------------------------------

class HurstEstimator:
    """
    Estimates the Hurst exponent using the rescaled range (R/S) method.
    H > 0.5 = trending, H < 0.5 = mean-reverting.
    """

    def __init__(self, min_window: int = 32, max_window: int = 128):
        self.min_window = min_window
        self.max_window = max_window

    def estimate(self, prices: np.ndarray) -> float:
        """Returns Hurst exponent for a price series."""
        log_prices = np.log(prices + 1e-12)
        returns = np.diff(log_prices)
        n = len(returns)
        if n < self.min_window:
            return 0.5

        lags = []
        rs_values = []
        for lag in range(self.min_window, min(n // 2, self.max_window), 8):
            rs = self._rs_stat(returns[:lag])
            if rs > 0:
                lags.append(np.log(lag))
                rs_values.append(np.log(rs))

        if len(lags) < 4:
            return 0.5

        # Linear regression log(RS) ~ H * log(n)
        lags_arr = np.array(lags)
        rs_arr = np.array(rs_values)
        H = np.polyfit(lags_arr, rs_arr, 1)[0]
        return float(np.clip(H, 0.0, 1.0))

    def _rs_stat(self, returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        mean_r = np.mean(returns)
        deviations = np.cumsum(returns - mean_r)
        R = np.max(deviations) - np.min(deviations)
        S = np.std(returns, ddof=1)
        if S < 1e-12:
            return 0.0
        return R / S

    def classify(self, H: float) -> HurstRegime:
        if H > 0.6:
            return HurstRegime.TRENDING
        elif H < 0.4:
            return HurstRegime.MEAN_REVERTING
        else:
            return HurstRegime.RANDOM_WALK


# ---------------------------------------------------------------------------
# CF Cross (Centrifugal Force crossover) detector
# ---------------------------------------------------------------------------

class CFCrossDetector:
    """
    Detects crossovers between a fast and slow exponential moving average
    to simulate the "CF cross" signal in the LARSA framework.

    The fast EMA represents inertial momentum and the slow EMA the
    gravitational equilibrium. A bullish cross means momentum has overcome
    the equilibrium (BH escape velocity condition).
    """

    def __init__(self, fast_period: int = 8, slow_period: int = 21):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._ema_fast: Dict[str, float] = {}
        self._ema_slow: Dict[str, float] = {}
        self._prev_cross: Dict[str, float] = {}
        self._alpha_fast = 2.0 / (fast_period + 1)
        self._alpha_slow = 2.0 / (slow_period + 1)

    def update(self, symbol: str, price: float) -> float:
        """
        Update EMAs and return the cross value (fast - slow, normalized).
        Positive = bullish, negative = bearish.
        """
        alpha_f = self._alpha_fast
        alpha_s = self._alpha_slow

        if symbol not in self._ema_fast:
            self._ema_fast[symbol] = price
            self._ema_slow[symbol] = price
            self._prev_cross[symbol] = 0.0
            return 0.0

        ema_f = alpha_f * price + (1 - alpha_f) * self._ema_fast[symbol]
        ema_s = alpha_s * price + (1 - alpha_s) * self._ema_slow[symbol]
        self._ema_fast[symbol] = ema_f
        self._ema_slow[symbol] = ema_s

        cross = (ema_f - ema_s) / (ema_s + 1e-12)
        self._prev_cross[symbol] = cross
        return cross

    def get_cross(self, symbol: str) -> float:
        return self._prev_cross.get(symbol, 0.0)

    def compute_from_series(self, prices: np.ndarray) -> np.ndarray:
        """Batch compute CF cross for a numpy price array."""
        n = len(prices)
        ema_f = np.zeros(n)
        ema_s = np.zeros(n)
        ema_f[0] = ema_s[0] = prices[0]
        for i in range(1, n):
            ema_f[i] = self._alpha_fast * prices[i] + (1 - self._alpha_fast) * ema_f[i - 1]
            ema_s[i] = self._alpha_slow * prices[i] + (1 - self._alpha_slow) * ema_s[i - 1]
        return (ema_f - ema_s) / (ema_s + 1e-12)


# ---------------------------------------------------------------------------
# GARCH(1,1) volatility estimator
# ---------------------------------------------------------------------------

class GARCHVolEstimator:
    """
    GARCH(1,1) volatility estimator using iterative updates.
    omega + alpha + beta < 1 ensures stationarity.
    """

    def __init__(
        self,
        omega: float = 1e-6,
        alpha: float = 0.1,
        beta: float = 0.85,
        init_var: float = 0.0004,  # initial variance (0.02^2)
    ):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self._variance: Dict[str, float] = {}
        self.init_var = init_var

    def update(self, symbol: str, return_: float) -> float:
        """Update GARCH variance and return annualized vol."""
        prev_var = self._variance.get(symbol, self.init_var)
        new_var = self.omega + self.alpha * return_**2 + self.beta * prev_var
        new_var = max(new_var, 1e-10)
        self._variance[symbol] = new_var
        # Annualize: 26 bars/day * 252 trading days
        ann_vol = np.sqrt(new_var * 26 * 252)
        return float(ann_vol)

    def compute_from_series(self, returns: np.ndarray) -> np.ndarray:
        """Batch GARCH volatility for an array of returns."""
        n = len(returns)
        variances = np.zeros(n)
        variances[0] = self.init_var
        for i in range(1, n):
            variances[i] = self.omega + self.alpha * returns[i - 1]**2 + self.beta * variances[i - 1]
        return np.sqrt(variances * 26 * 252)

    def get_vol(self, symbol: str) -> float:
        var = self._variance.get(symbol, self.init_var)
        return float(np.sqrt(var * 26 * 252))


# ---------------------------------------------------------------------------
# Quaternion Navigation (heading direction smoothing)
# ---------------------------------------------------------------------------

class QuaternionNavigator:
    """
    Uses quaternion-like rotation to smooth the direction vector
    of price movement. Represents direction as a unit quaternion
    in (price_momentum, vol_adjusted_momentum) 2D space, providing
    smooth heading transitions without gimbal lock.
    """

    def __init__(self, smoothing: float = 0.1):
        self.smoothing = smoothing
        # Quaternion components (q0=scalar, q1, q2 = vector components)
        self._q: Dict[str, np.ndarray] = {}

    def update(self, symbol: str, price_mom: float, vol_adj_mom: float) -> Tuple[float, float]:
        """
        Update quaternion state and return smoothed (heading, magnitude).
        heading: angle in [-pi, pi] from the positive price axis.
        magnitude: strength of the directional signal.
        """
        # New direction vector
        raw = np.array([price_mom, vol_adj_mom])
        mag = np.linalg.norm(raw)
        if mag < 1e-12:
            raw_unit = np.array([0.0, 0.0])
        else:
            raw_unit = raw / mag

        # Represent as quaternion: q = [cos(theta/2), sin(theta/2)*axis]
        theta = np.arctan2(raw_unit[1], raw_unit[0]) if mag > 0 else 0.0
        new_q = np.array([np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0])

        if symbol not in self._q:
            self._q[symbol] = new_q
        else:
            # Slerp (spherical linear interpolation)
            q_old = self._q[symbol]
            dot = np.clip(np.dot(q_old, new_q), -1.0, 1.0)
            if dot < 0:
                new_q = -new_q
                dot = -dot
            t = self.smoothing
            if 1 - dot > 1e-6:
                omega_val = np.arccos(dot)
                s0 = np.sin((1 - t) * omega_val) / np.sin(omega_val)
                s1 = np.sin(t * omega_val) / np.sin(omega_val)
                interp_q = s0 * q_old + s1 * new_q
            else:
                interp_q = (1 - t) * q_old + t * new_q
            norm = np.linalg.norm(interp_q)
            self._q[symbol] = interp_q / (norm + 1e-12)

        # Extract heading from quaternion
        q = self._q[symbol]
        heading = 2 * np.arctan2(q[1], q[0])
        return float(heading), float(mag)

    def get_heading(self, symbol: str) -> float:
        q = self._q.get(symbol, np.array([1.0, 0.0, 0.0, 0.0]))
        return float(2 * np.arctan2(q[1], q[0]))


# ---------------------------------------------------------------------------
# Position Sizer (Kelly with vol targeting)
# ---------------------------------------------------------------------------

class PositionSizer:
    """
    Kelly-based position sizing with volatility targeting.

    Kelly fraction: f* = (mu - r) / sigma^2
    Vol-targeted: weight = target_vol / asset_vol

    Final size = min(kelly_fraction, vol_target_weight) * signal_strength
    """

    def __init__(
        self,
        target_annual_vol: float = 0.15,   # 15% target portfolio vol
        max_position_weight: float = 0.40, # max 40% in any single asset
        kelly_fraction: float = 0.25,      # fractional Kelly (25% of full Kelly)
        min_signal_threshold: float = 0.05,
    ):
        self.target_annual_vol = target_annual_vol
        self.max_position_weight = max_position_weight
        self.kelly_fraction = kelly_fraction
        self.min_signal_threshold = min_signal_threshold

    def compute_weight(
        self,
        signal_strength: float,
        asset_vol: float,
        expected_return: float,
        portfolio_equity: float,
        current_price: float,
    ) -> float:
        """
        Compute target portfolio weight for the asset.
        Returns a value in [-max_position_weight, max_position_weight].
        """
        if abs(signal_strength) < self.min_signal_threshold:
            return 0.0
        if asset_vol < 1e-6:
            return 0.0

        # Volatility-targeting weight
        bars_per_year = 26 * 252
        bar_vol = asset_vol / np.sqrt(bars_per_year)
        target_bar_vol = self.target_annual_vol / np.sqrt(bars_per_year)
        vol_weight = target_bar_vol / (bar_vol + 1e-12)

        # Kelly weight (simplified: win_rate is implied by expected_return)
        kelly_w = self.kelly_fraction * expected_return / (asset_vol**2 + 1e-12)

        # Combine: take min of the two
        raw_weight = min(abs(vol_weight), abs(kelly_w)) * np.sign(signal_strength)
        clipped = np.clip(raw_weight, -self.max_position_weight, self.max_position_weight)
        return float(clipped)

    def weight_to_quantity(
        self,
        target_weight: float,
        portfolio_equity: float,
        current_price: float,
    ) -> float:
        """Convert a portfolio weight to a share/coin quantity."""
        if current_price <= 0:
            return 0.0
        target_notional = target_weight * portfolio_equity
        return target_notional / current_price


# ---------------------------------------------------------------------------
# Signal Adapter: translates LARSA indicators to SignalEvents
# ---------------------------------------------------------------------------

class SignalAdapter:
    """
    Aggregates BH mass, CF cross, Hurst, and GARCH signals into a
    unified SignalEvent with direction and strength.

    Signal combination logic:
      - If BH_ACTIVE and CF cross positive and Hurst is TRENDING: LONG
      - If BH_ACTIVE and CF cross negative and Hurst is TRENDING: SHORT
      - If BH_INACTIVE or Hurst is MEAN_REVERTING: fade extreme moves
      - Strength = weighted combination of sub-signals
    """

    def __init__(
        self,
        bh_weight: float = 0.4,
        cf_weight: float = 0.35,
        hurst_weight: float = 0.25,
        min_strength: float = 0.1,
    ):
        self.bh_weight = bh_weight
        self.cf_weight = cf_weight
        self.hurst_weight = hurst_weight
        self.min_strength = min_strength

    def compute_signal(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        bh_mass: float,
        bh_regime: BHRegime,
        cf_cross: float,
        hurst: float,
        hurst_regime: HurstRegime,
        garch_vol: float,
        nav_heading: float,
        mass_threshold: float = 0.0025,
    ) -> Optional[SignalEvent]:
        """
        Combine sub-signals into a SignalEvent.
        Returns None if combined strength is below the minimum threshold.
        """
        # Normalize BH mass contribution
        bh_norm = float(np.tanh(bh_mass / (mass_threshold + 1e-12)))

        # CF cross contribution (already normalized)
        cf_norm = float(np.tanh(cf_cross * 50))  # amplify small CF values

        # Hurst contribution: trending -> amplify, mean-reverting -> reduce
        if hurst_regime == HurstRegime.TRENDING:
            hurst_factor = (hurst - 0.5) * 2  # 0 to 1
        elif hurst_regime == HurstRegime.MEAN_REVERTING:
            hurst_factor = (hurst - 0.5) * 2  # -1 to 0 (negative)
        else:
            hurst_factor = 0.0

        # Navigation heading contribution
        nav_norm = float(np.sin(nav_heading))  # heading in [-pi, pi] -> [-1, 1]

        # Weighted combination
        if bh_regime == BHRegime.BH_ACTIVE:
            bh_contrib = self.bh_weight * bh_norm
        elif bh_regime == BHRegime.TRANSITIONING:
            bh_contrib = self.bh_weight * bh_norm * 0.5
        else:
            bh_contrib = 0.0

        cf_contrib = self.cf_weight * cf_norm
        hurst_contrib = self.hurst_weight * hurst_factor
        nav_contrib = 0.1 * nav_norm  # small nav weight

        raw_strength = bh_contrib + cf_contrib + hurst_contrib + nav_contrib

        # In mean-reverting regime, flip the signal direction
        if hurst_regime == HurstRegime.MEAN_REVERTING:
            raw_strength = -raw_strength * 0.5  # fade moves, reduced size

        # Determine direction
        if abs(raw_strength) < self.min_strength:
            return None

        if raw_strength > 0:
            direction = Direction.LONG
        else:
            direction = Direction.SHORT

        return SignalEvent(
            event_type=EventType.SIGNAL,
            timestamp=timestamp,
            symbol=symbol,
            direction=direction,
            strength=float(np.clip(raw_strength, -1.0, 1.0)),
            bh_mass=bh_mass,
            cf_cross=cf_cross,
            hurst=hurst,
            garch_vol=garch_vol,
            regime=bh_regime.value,
        )


# ---------------------------------------------------------------------------
# Main LARSA Strategy Adapter
# ---------------------------------------------------------------------------

class LARSAStrategyAdapter:
    """
    Full LARSA strategy adapter.

    Wraps all sub-components (BH mass, CF cross, Hurst, GARCH, Nav) and
    generates SignalEvents on each MarketEvent.

    Minimum warmup: 128 bars (for Hurst estimation).
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = "15m",
        bh_params: Optional[Dict] = None,
        cf_params: Optional[Dict] = None,
        hurst_params: Optional[Dict] = None,
        garch_params: Optional[Dict] = None,
        sizer_params: Optional[Dict] = None,
        warmup_bars: int = 128,
    ):
        self.symbols = symbols or []
        self.timeframe = timeframe
        self.warmup_bars = warmup_bars

        self.bh_adapter = BHMassAdapter(**(bh_params or {}))
        self.cf_detector = CFCrossDetector(**(cf_params or {}))
        self.hurst_estimator = HurstEstimator(**(hurst_params or {}))
        self.garch_estimator = GARCHVolEstimator(**(garch_params or {}))
        self.quat_nav = QuaternionNavigator()
        self.signal_adapter = SignalAdapter()
        self.sizer = PositionSizer(**(sizer_params or {}))

        self._data = MultiAssetData()
        self._bar_counts: Dict[str, int] = {}
        self._last_signals: Dict[str, Optional[SignalEvent]] = {}

        # Callbacks for generated signals (set by engine)
        self._signal_callbacks: List[Callable] = []

    def register_signal_callback(self, fn: Callable) -> None:
        self._signal_callbacks.append(fn)

    def on_market_event(self, event: MarketEvent) -> Optional[List[SignalEvent]]:
        """
        Process a MarketEvent and potentially emit a SignalEvent.
        Returns list of SignalEvents (may be empty).
        """
        if not isinstance(event, MarketEvent):
            return None
        if event.timeframe != self.timeframe:
            return None

        symbol = event.symbol
        self._data.update(event)
        self._bar_counts[symbol] = self._bar_counts.get(symbol, 0) + 1

        buf = self._data.get_buffer(symbol, self.timeframe)
        if buf is None or not buf.is_ready(self.warmup_bars):
            return None

        signals = self._compute_signals(symbol, event, buf)
        for sig in signals:
            self._last_signals[symbol] = sig
            for cb in self._signal_callbacks:
                cb(sig)

        return signals if signals else None

    def _compute_signals(
        self,
        symbol: str,
        event: MarketEvent,
        buf: BarBuffer,
    ) -> List[SignalEvent]:
        """Run all sub-models and combine into signal(s)."""

        # 1. BH mass
        bh_mass = self.bh_adapter.compute_mass(symbol, buf)
        bh_regime = self.bh_adapter.get_regime(symbol)

        # 2. CF cross
        cf_cross = self.cf_detector.update(symbol, event.close)

        # 3. Hurst exponent (expensive -- computed every N bars)
        bar_count = self._bar_counts.get(symbol, 0)
        if bar_count % 8 == 0 or symbol not in self._last_signals:
            closes = buf.closes(128)
            hurst = self.hurst_estimator.estimate(closes)
        else:
            # Use cached hurst from last signal
            last = self._last_signals.get(symbol)
            hurst = last.hurst if last is not None else 0.5

        hurst_regime = self.hurst_estimator.classify(hurst)

        # 4. GARCH volatility
        closes = buf.closes(2)
        if len(closes) >= 2:
            ret = np.log(closes[-1] / (closes[-2] + 1e-12))
            garch_vol = self.garch_estimator.update(symbol, ret)
        else:
            garch_vol = self.garch_estimator.get_vol(symbol)

        # 5. Quaternion navigation
        price_mom = cf_cross  # use CF cross as price momentum proxy
        vol_adj_mom = cf_cross / (garch_vol + 1e-12)
        nav_heading, nav_mag = self.quat_nav.update(symbol, price_mom, vol_adj_mom)

        # 6. Combine into signal
        sig = self.signal_adapter.compute_signal(
            symbol=symbol,
            timestamp=event.timestamp,
            bh_mass=bh_mass,
            bh_regime=bh_regime,
            cf_cross=cf_cross,
            hurst=hurst,
            hurst_regime=hurst_regime,
            garch_vol=garch_vol,
            nav_heading=nav_heading,
            mass_threshold=self.bh_adapter.mass_threshold,
        )

        if sig is not None:
            sig.target_weight = self._compute_target_weight(sig, garch_vol)
            return [sig]
        return []

    def _compute_target_weight(self, sig: SignalEvent, garch_vol: float) -> float:
        """Compute Kelly/vol-targeted weight for the signal."""
        expected_return = abs(sig.strength) * 0.001  # rough proxy
        weight = self.sizer.compute_weight(
            signal_strength=sig.strength,
            asset_vol=garch_vol,
            expected_return=expected_return,
            portfolio_equity=1.0,
            current_price=1.0,
        )
        return weight

    def get_last_signal(self, symbol: str) -> Optional[SignalEvent]:
        return self._last_signals.get(symbol)

    def get_bh_mass(self, symbol: str) -> float:
        return self.bh_adapter._mass.get(symbol, 0.0)

    def get_hurst(self, symbol: str) -> float:
        last = self._last_signals.get(symbol)
        return last.hurst if last is not None else 0.5

    def reset(self) -> None:
        self.bh_adapter.reset()
        self._bar_counts.clear()
        self._last_signals.clear()
        self._data = MultiAssetData()
