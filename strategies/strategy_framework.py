"""
strategy_framework.py -- Unified strategy framework with multiple strategy
implementations, risk overlay, multi-strategy blending, and performance tracking.

All numerics via numpy/scipy.  No pandas dependency.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import minimize
from scipy.linalg import solve

FloatArray = NDArray[np.float64]


# ===================================================================
# 1.  Base classes and enums
# ===================================================================

class Side(enum.IntEnum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class Signal:
    """Cross-sectional signal for a single period."""
    timestamp: int
    values: FloatArray          # (n_assets,) raw signal values
    confidence: FloatArray | None = None  # (n_assets,) optional confidence


@dataclass
class Position:
    """Portfolio position vector."""
    weights: FloatArray         # (n_assets,) signed weights
    notional: float = 1.0
    leverage: float = 1.0

    @property
    def gross_exposure(self) -> float:
        return float(np.abs(self.weights).sum())

    @property
    def net_exposure(self) -> float:
        return float(self.weights.sum())

    @property
    def long_exposure(self) -> float:
        return float(self.weights[self.weights > 0].sum())

    @property
    def short_exposure(self) -> float:
        return float(np.abs(self.weights[self.weights < 0]).sum())


@dataclass
class TradeOrder:
    timestamp: int
    asset_idx: int
    target_weight: float
    current_weight: float
    side: Side
    urgency: float = 0.5


@dataclass
class StrategyState:
    """Mutable state carried between periods."""
    positions: Position
    cash: float = 0.0
    pnl_history: List[float] = field(default_factory=list)
    signal_history: List[Signal] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyBase(abc.ABC):
    """Abstract base class for all strategies."""

    def __init__(self, n_assets: int, name: str = "base"):
        self.n_assets = n_assets
        self.name = name
        self.state = StrategyState(
            positions=Position(weights=np.zeros(n_assets))
        )

    @abc.abstractmethod
    def generate_signals(
        self, prices: FloatArray, volumes: FloatArray, t: int, **kwargs: Any
    ) -> Signal:
        """Generate raw signal for current period."""
        ...

    @abc.abstractmethod
    def size_positions(self, signal: Signal, **kwargs: Any) -> Position:
        """Convert signal to target position."""
        ...

    def manage_risk(self, position: Position, prices: FloatArray, t: int) -> Position:
        """Apply risk management rules. Override for custom logic."""
        return position

    def step(
        self, prices: FloatArray, volumes: FloatArray, t: int, **kwargs: Any
    ) -> Position:
        """Full step: signal -> sizing -> risk."""
        signal = self.generate_signals(prices, volumes, t, **kwargs)
        self.state.signal_history.append(signal)
        position = self.size_positions(signal, **kwargs)
        position = self.manage_risk(position, prices, t)
        self.state.positions = position
        return position

    def reset(self) -> None:
        self.state = StrategyState(
            positions=Position(weights=np.zeros(self.n_assets))
        )


# ===================================================================
# 2.  Momentum Strategy
# ===================================================================

@dataclass
class MomentumConfig:
    lookback: int = 252
    skip: int = 21                       # skip most recent month
    vol_lookback: int = 63
    vol_target: float = 0.10
    n_long: int | None = None            # top N; None = top quintile
    n_short: int | None = None
    cross_sectional: bool = True


class MomentumStrategy(StrategyBase):
    """Cross-sectional momentum with volatility scaling."""

    def __init__(self, n_assets: int, config: MomentumConfig | None = None):
        super().__init__(n_assets, name="momentum")
        self.cfg = config or MomentumConfig()
        self._price_buffer: List[FloatArray] = []

    def generate_signals(
        self, prices: FloatArray, volumes: FloatArray, t: int, **kwargs: Any
    ) -> Signal:
        self._price_buffer.append(prices.copy())
        buf = self._price_buffer
        lb = self.cfg.lookback + self.cfg.skip
        if len(buf) < lb:
            return Signal(timestamp=t, values=np.zeros(self.n_assets))
        p_start = buf[-lb]
        p_end = buf[-(self.cfg.skip + 1)]
        mom = p_end / (p_start + 1e-12) - 1.0
        return Signal(timestamp=t, values=mom)

    def size_positions(self, signal: Signal, **kwargs: Any) -> Position:
        s = signal.values
        n = self.n_assets
        n_long = self.cfg.n_long or max(n // 5, 1)
        n_short = self.cfg.n_short or max(n // 5, 1)

        if self.cfg.cross_sectional:
            ranks = stats.rankdata(s)
            long_mask = ranks > (n - n_long)
            short_mask = ranks <= n_short
            w = np.zeros(n)
            w[long_mask] = 1.0 / max(long_mask.sum(), 1)
            w[short_mask] = -1.0 / max(short_mask.sum(), 1)
        else:
            w = s / (np.abs(s).sum() + 1e-12)

        # Vol scaling
        w = self._vol_scale(w)
        return Position(weights=w)

    def _vol_scale(self, weights: FloatArray) -> FloatArray:
        buf = self._price_buffer
        vl = self.cfg.vol_lookback
        if len(buf) < vl + 1:
            return weights
        prices_arr = np.array(buf[-vl - 1 :])
        rets = np.diff(np.log(prices_arr + 1e-12), axis=0)
        port_ret = rets @ weights
        realized_vol = port_ret.std() * np.sqrt(252)
        if realized_vol > 1e-6:
            scale = self.cfg.vol_target / realized_vol
            weights = weights * min(scale, 3.0)
        return weights


# ===================================================================
# 3.  Mean Reversion Strategy
# ===================================================================

@dataclass
class MeanReversionConfig:
    lookback: int = 21
    entry_z: float = 2.0
    exit_z: float = 0.5
    vol_lookback: int = 63
    vol_target: float = 0.10
    ou_calibration: bool = True


class MeanReversionStrategy(StrategyBase):
    """Z-score based mean reversion with optional OU calibration."""

    def __init__(self, n_assets: int, config: MeanReversionConfig | None = None):
        super().__init__(n_assets, name="mean_reversion")
        self.cfg = config or MeanReversionConfig()
        self._price_buffer: List[FloatArray] = []
        self._ou_params: Dict[int, Tuple[float, float, float]] = {}

    def generate_signals(
        self, prices: FloatArray, volumes: FloatArray, t: int, **kwargs: Any
    ) -> Signal:
        self._price_buffer.append(prices.copy())
        lb = self.cfg.lookback
        if len(self._price_buffer) < lb + 1:
            return Signal(timestamp=t, values=np.zeros(self.n_assets))
        window = np.array(self._price_buffer[-lb - 1 :])
        log_p = np.log(window + 1e-12)
        mu = log_p.mean(axis=0)
        std = log_p.std(axis=0) + 1e-12
        z = (log_p[-1] - mu) / std
        # Negative z = signal to buy (mean reversion)
        return Signal(timestamp=t, values=-z)

    def size_positions(self, signal: Signal, **kwargs: Any) -> Position:
        z = -signal.values  # positive z = overbought
        w = np.zeros(self.n_assets)
        for i in range(self.n_assets):
            if z[i] > self.cfg.entry_z:
                w[i] = -1.0  # short overbought
            elif z[i] < -self.cfg.entry_z:
                w[i] = 1.0   # long oversold
            elif abs(z[i]) < self.cfg.exit_z:
                w[i] = 0.0   # exit near mean
            else:
                w[i] = self.state.positions.weights[i]  # hold

        # Normalize
        gross = np.abs(w).sum()
        if gross > 0:
            w = w / gross

        # OU calibration scaling
        if self.cfg.ou_calibration and len(self._price_buffer) > 100:
            for i in range(self.n_assets):
                kappa, mu_ou, sigma_ou = self._calibrate_ou(i)
                self._ou_params[i] = (kappa, mu_ou, sigma_ou)
                if kappa > 0:
                    halflife = np.log(2) / kappa
                    # More weight to faster mean-reverting
                    w[i] *= min(self.cfg.lookback / (halflife + 1e-6), 3.0)

        # Vol scale
        w = self._vol_scale(w)
        return Position(weights=w)

    def _calibrate_ou(self, asset_idx: int) -> Tuple[float, float, float]:
        """MLE for Ornstein-Uhlenbeck: dx = kappa*(mu - x)dt + sigma*dW."""
        prices_arr = np.array(self._price_buffer[-100:])
        x = np.log(prices_arr[:, asset_idx] + 1e-12)
        n = len(x) - 1
        if n < 10:
            return (1.0, x[-1], 0.01)
        dt = 1.0 / 252.0
        x0 = x[:-1]
        x1 = x[1:]
        Sx = x0.sum()
        Sy = x1.sum()
        Sxx = (x0 ** 2).sum()
        Sxy = (x0 * x1).sum()
        Syy = (x1 ** 2).sum()
        denom = n * Sxx - Sx ** 2
        if abs(denom) < 1e-12:
            return (1.0, x[-1], 0.01)
        a = (Sy * Sxx - Sx * Sxy) / denom
        b = (n * Sxy - Sx * Sy) / denom
        if b >= 1.0 or b <= 0:
            return (1.0, x[-1], 0.01)
        kappa = -np.log(b) / dt
        mu_ou = a / (1.0 - b)
        residuals = x1 - a - b * x0
        sigma_ou = residuals.std() / np.sqrt(dt)
        return (float(kappa), float(mu_ou), float(sigma_ou))

    def _vol_scale(self, weights: FloatArray) -> FloatArray:
        if len(self._price_buffer) < self.cfg.vol_lookback + 1:
            return weights
        arr = np.array(self._price_buffer[-self.cfg.vol_lookback - 1 :])
        rets = np.diff(np.log(arr + 1e-12), axis=0)
        port_ret = rets @ weights
        rv = port_ret.std() * np.sqrt(252)
        if rv > 1e-6:
            weights *= min(self.cfg.vol_target / rv, 3.0)
        return weights


# ===================================================================
# 4.  Stat Arb / Pairs Trading Strategy
# ===================================================================

@dataclass
class StatArbConfig:
    lookback: int = 63
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    use_kalman: bool = True
    kalman_delta: float = 1e-4
    kalman_ve: float = 1e-3


class StatArbStrategy(StrategyBase):
    """Pairs trading with dynamic hedge ratio via Kalman filter."""

    def __init__(self, n_assets: int, config: StatArbConfig | None = None):
        super().__init__(n_assets, name="stat_arb")
        self.cfg = config or StatArbConfig()
        self._price_buffer: List[FloatArray] = []
        # Kalman state per pair (we pair consecutive assets)
        self.n_pairs = n_assets // 2
        self._kalman_states: List[Dict[str, Any]] = []
        for _ in range(self.n_pairs):
            self._kalman_states.append({
                "beta": 1.0,
                "P": 1.0,
                "R": self.cfg.kalman_ve,
                "Q": self.cfg.kalman_delta,
            })

    def generate_signals(
        self, prices: FloatArray, volumes: FloatArray, t: int, **kwargs: Any
    ) -> Signal:
        self._price_buffer.append(prices.copy())
        signals = np.zeros(self.n_assets)
        lb = self.cfg.lookback
        if len(self._price_buffer) < lb:
            return Signal(timestamp=t, values=signals)

        for pair_idx in range(self.n_pairs):
            i = pair_idx * 2
            j = pair_idx * 2 + 1
            p_i = prices[i]
            p_j = prices[j]

            if self.cfg.use_kalman:
                beta = self._kalman_update(pair_idx, p_i, p_j)
            else:
                arr = np.array(self._price_buffer[-lb:])
                beta = self._ols_hedge(arr[:, i], arr[:, j])

            spread = np.log(p_i + 1e-12) - beta * np.log(p_j + 1e-12)
            # Z-score of spread
            spread_hist = []
            for k in range(max(0, len(self._price_buffer) - lb), len(self._price_buffer)):
                pi_k = self._price_buffer[k][i]
                pj_k = self._price_buffer[k][j]
                spread_hist.append(np.log(pi_k + 1e-12) - beta * np.log(pj_k + 1e-12))
            sh = np.array(spread_hist)
            z = (spread - sh.mean()) / (sh.std() + 1e-12)
            signals[i] = -z   # sell i when spread is high
            signals[j] = z * beta

        return Signal(timestamp=t, values=signals)

    def _kalman_update(self, pair_idx: int, p_i: float, p_j: float) -> float:
        ks = self._kalman_states[pair_idx]
        x = np.log(p_j + 1e-12)
        y = np.log(p_i + 1e-12)
        # Predict
        beta_pred = ks["beta"]
        P_pred = ks["P"] + ks["Q"]
        # Update
        e = y - beta_pred * x
        S = x ** 2 * P_pred + ks["R"]
        K = P_pred * x / (S + 1e-12)
        ks["beta"] = beta_pred + K * e
        ks["P"] = (1 - K * x) * P_pred
        return ks["beta"]

    def _ols_hedge(self, y: FloatArray, x: FloatArray) -> float:
        log_x = np.log(x + 1e-12)
        log_y = np.log(y + 1e-12)
        X = np.column_stack([np.ones(len(log_x)), log_x])
        beta = np.linalg.lstsq(X, log_y, rcond=None)[0]
        return float(beta[1])

    def size_positions(self, signal: Signal, **kwargs: Any) -> Position:
        s = signal.values
        w = np.zeros(self.n_assets)
        for pair_idx in range(self.n_pairs):
            i = pair_idx * 2
            j = pair_idx * 2 + 1
            z_i = abs(s[i])
            if z_i > self.cfg.entry_z:
                w[i] = np.sign(s[i]) / self.n_pairs
                w[j] = np.sign(s[j]) / self.n_pairs
            elif z_i < self.cfg.exit_z:
                w[i] = 0.0
                w[j] = 0.0
            elif z_i > self.cfg.stop_z:
                w[i] = 0.0
                w[j] = 0.0
            else:
                w[i] = self.state.positions.weights[i]
                w[j] = self.state.positions.weights[j]
        return Position(weights=w)


# ===================================================================
# 5.  Breakout Strategy
# ===================================================================

@dataclass
class BreakoutConfig:
    channel_lookback: int = 20           # Donchian channel
    atr_lookback: int = 14
    atr_stop_mult: float = 2.0
    vol_target: float = 0.10


class BreakoutStrategy(StrategyBase):
    """Donchian channel breakout with ATR-based stops."""

    def __init__(self, n_assets: int, config: BreakoutConfig | None = None):
        super().__init__(n_assets, name="breakout")
        self.cfg = config or BreakoutConfig()
        self._price_buffer: List[FloatArray] = []
        self._high_buffer: List[FloatArray] = []
        self._low_buffer: List[FloatArray] = []
        self._entry_prices: FloatArray = np.zeros(n_assets)
        self._stop_prices: FloatArray = np.zeros(n_assets)

    def generate_signals(
        self, prices: FloatArray, volumes: FloatArray, t: int, **kwargs: Any
    ) -> Signal:
        high = kwargs.get("high", prices * 1.005)
        low = kwargs.get("low", prices * 0.995)
        self._price_buffer.append(prices.copy())
        self._high_buffer.append(high.copy())
        self._low_buffer.append(low.copy())
        lb = self.cfg.channel_lookback
        if len(self._price_buffer) < lb:
            return Signal(timestamp=t, values=np.zeros(self.n_assets))

        highs = np.array(self._high_buffer[-lb:])
        lows = np.array(self._low_buffer[-lb:])
        upper = highs.max(axis=0)
        lower = lows.min(axis=0)
        mid = (upper + lower) / 2.0

        signals = np.zeros(self.n_assets)
        for i in range(self.n_assets):
            if prices[i] > upper[i]:
                signals[i] = 1.0    # long breakout
            elif prices[i] < lower[i]:
                signals[i] = -1.0   # short breakout
            else:
                signals[i] = 0.0
        return Signal(timestamp=t, values=signals)

    def size_positions(self, signal: Signal, **kwargs: Any) -> Position:
        s = signal.values
        atr = self._compute_atr()
        w = np.zeros(self.n_assets)
        for i in range(self.n_assets):
            if s[i] != 0:
                # Size inversely proportional to ATR
                if atr[i] > 1e-8:
                    risk_per_unit = atr[i] * self.cfg.atr_stop_mult
                    w[i] = s[i] * self.cfg.vol_target / (risk_per_unit * np.sqrt(252) + 1e-12)
                    self._entry_prices[i] = self._price_buffer[-1][i]
                    self._stop_prices[i] = self._entry_prices[i] - s[i] * risk_per_unit
            else:
                w[i] = self.state.positions.weights[i]
        # Cap leverage
        gross = np.abs(w).sum()
        if gross > 3.0:
            w *= 3.0 / gross
        return Position(weights=w)

    def _compute_atr(self) -> FloatArray:
        lb = self.cfg.atr_lookback
        if len(self._price_buffer) < lb + 1:
            return np.ones(self.n_assets) * 0.02
        prices = np.array(self._price_buffer[-lb - 1 :])
        highs = np.array(self._high_buffer[-lb:]) if len(self._high_buffer) >= lb else prices[1:]
        lows = np.array(self._low_buffer[-lb:]) if len(self._low_buffer) >= lb else prices[1:]
        tr = np.maximum(
            highs - lows,
            np.maximum(
                np.abs(highs - prices[:-1]),
                np.abs(lows - prices[:-1]),
            ),
        )
        return tr.mean(axis=0)

    def manage_risk(self, position: Position, prices: FloatArray, t: int) -> Position:
        w = position.weights.copy()
        for i in range(self.n_assets):
            if w[i] > 0 and prices[i] < self._stop_prices[i]:
                w[i] = 0.0
            elif w[i] < 0 and prices[i] > self._stop_prices[i]:
                w[i] = 0.0
        return Position(weights=w)


# ===================================================================
# 6.  Carry Strategy
# ===================================================================

@dataclass
class CarryConfig:
    vol_target: float = 0.10
    lookback: int = 63


class CarryStrategy(StrategyBase):
    """Carry strategy: yield curve, FX, or funding rate carry."""

    def __init__(self, n_assets: int, config: CarryConfig | None = None):
        super().__init__(n_assets, name="carry")
        self.cfg = config or CarryConfig()
        self._price_buffer: List[FloatArray] = []

    def generate_signals(
        self, prices: FloatArray, volumes: FloatArray, t: int, **kwargs: Any
    ) -> Signal:
        carry_rates = kwargs.get("carry_rates", None)
        self._price_buffer.append(prices.copy())
        if carry_rates is None:
            # Estimate carry from roll yield: (spot - future) / spot proxy
            if len(self._price_buffer) < 22:
                return Signal(timestamp=t, values=np.zeros(self.n_assets))
            spot = prices
            fut_proxy = np.array(self._price_buffer[-22])
            carry_est = (spot - fut_proxy) / (fut_proxy + 1e-12)
            return Signal(timestamp=t, values=carry_est)
        return Signal(timestamp=t, values=carry_rates)

    def size_positions(self, signal: Signal, **kwargs: Any) -> Position:
        s = signal.values
        # Rank-based sizing
        ranks = stats.rankdata(s)
        n = self.n_assets
        w = (ranks - (n + 1) / 2.0) / n
        # Vol scaling
        if len(self._price_buffer) > self.cfg.lookback:
            arr = np.array(self._price_buffer[-self.cfg.lookback - 1 :])
            rets = np.diff(np.log(arr + 1e-12), axis=0)
            port_ret = rets @ w
            rv = port_ret.std() * np.sqrt(252)
            if rv > 1e-6:
                w *= min(self.cfg.vol_target / rv, 3.0)
        return Position(weights=w)


# ===================================================================
# 7.  Volatility Strategy
# ===================================================================

@dataclass
class VolatilityStrategyConfig:
    short_vol_weight: float = 0.6
    long_vol_weight: float = 0.4
    vol_lookback: int = 21
    vol_target: float = 0.10


class VolatilityStrategy(StrategyBase):
    """Short vol (iron condor proxy) + long vol (straddle proxy)."""

    def __init__(self, n_assets: int, config: VolatilityStrategyConfig | None = None):
        super().__init__(n_assets, name="volatility")
        self.cfg = config or VolatilityStrategyConfig()
        self._price_buffer: List[FloatArray] = []

    def generate_signals(
        self, prices: FloatArray, volumes: FloatArray, t: int, **kwargs: Any
    ) -> Signal:
        self._price_buffer.append(prices.copy())
        implied_vol = kwargs.get("implied_vol", None)
        lb = self.cfg.vol_lookback
        if len(self._price_buffer) < lb + 1:
            return Signal(timestamp=t, values=np.zeros(self.n_assets))
        arr = np.array(self._price_buffer[-lb - 1 :])
        realized_vol = np.diff(np.log(arr + 1e-12), axis=0).std(axis=0) * np.sqrt(252)
        if implied_vol is None:
            implied_vol = realized_vol * 1.1  # proxy: IV typically > RV
        vol_premium = implied_vol - realized_vol
        return Signal(timestamp=t, values=vol_premium)

    def size_positions(self, signal: Signal, **kwargs: Any) -> Position:
        vol_premium = signal.values
        w = np.zeros(self.n_assets)
        for i in range(self.n_assets):
            if vol_premium[i] > 0:
                # Sell vol (short vol strategy)
                w[i] = -self.cfg.short_vol_weight * vol_premium[i] * 10
            else:
                # Buy vol (long vol strategy)
                w[i] = -self.cfg.long_vol_weight * vol_premium[i] * 10
        # Normalize
        gross = np.abs(w).sum()
        if gross > 0:
            w = w / gross * self.cfg.vol_target
        return Position(weights=w)


# ===================================================================
# 8.  Multi-Strategy Blender
# ===================================================================

@dataclass
class BlenderConfig:
    rebalance_frequency: int = 21
    regime_aware: bool = True
    vol_lookback: int = 63
    min_weight: float = 0.0
    max_weight: float = 1.0


class MultiStrategyBlender:
    """Regime-conditional strategy weight allocation."""

    def __init__(
        self,
        strategies: List[StrategyBase],
        config: BlenderConfig | None = None,
    ):
        self.strategies = strategies
        self.cfg = config or BlenderConfig()
        self.n_strategies = len(strategies)
        self.weights = np.ones(self.n_strategies) / self.n_strategies
        self._return_history: List[List[float]] = [[] for _ in strategies]
        self._price_buffer: List[FloatArray] = []

    def update_weights(self, t: int) -> None:
        if t % self.cfg.rebalance_frequency != 0:
            return
        if self.cfg.regime_aware:
            regime = self._detect_regime()
            self.weights = self._regime_weights(regime)
        else:
            self.weights = self._inverse_vol_weights()

    def _detect_regime(self) -> int:
        """Simple vol regime: 0=low, 1=medium, 2=high."""
        if len(self._price_buffer) < self.cfg.vol_lookback:
            return 1
        arr = np.array(self._price_buffer[-self.cfg.vol_lookback:])
        rets = np.diff(np.log(arr[:, 0] + 1e-12))
        vol = rets.std() * np.sqrt(252)
        if vol < 0.12:
            return 0
        elif vol < 0.25:
            return 1
        return 2

    def _regime_weights(self, regime: int) -> FloatArray:
        """Heuristic regime-dependent weights."""
        n = self.n_strategies
        w = np.ones(n) / n
        # Adjust based on strategy names
        for i, strat in enumerate(self.strategies):
            if regime == 0:  # low vol
                if "momentum" in strat.name:
                    w[i] *= 1.5
                if "mean_reversion" in strat.name:
                    w[i] *= 0.5
            elif regime == 2:  # high vol
                if "momentum" in strat.name:
                    w[i] *= 0.5
                if "mean_reversion" in strat.name:
                    w[i] *= 1.5
                if "volatility" in strat.name:
                    w[i] *= 1.3
        w = np.clip(w, self.cfg.min_weight, self.cfg.max_weight)
        w /= w.sum() + 1e-12
        return w

    def _inverse_vol_weights(self) -> FloatArray:
        """Weight inversely proportional to realized strategy vol."""
        n = self.n_strategies
        vols = np.ones(n)
        for i in range(n):
            if len(self._return_history[i]) > 20:
                vols[i] = np.std(self._return_history[i][-63:]) + 1e-8
        inv_vol = 1.0 / vols
        return inv_vol / inv_vol.sum()

    def step(
        self, prices: FloatArray, volumes: FloatArray, t: int, **kwargs: Any
    ) -> Position:
        self._price_buffer.append(prices.copy())
        self.update_weights(t)
        combined_w = np.zeros(len(prices))
        for i, strat in enumerate(self.strategies):
            pos = strat.step(prices, volumes, t, **kwargs)
            combined_w += self.weights[i] * pos.weights
            # Track returns
            if len(self._price_buffer) > 1:
                prev_p = self._price_buffer[-2]
                ret = np.log(prices / (prev_p + 1e-12))
                strat_ret = float(pos.weights @ ret)
                self._return_history[i].append(strat_ret)
        return Position(weights=combined_w)


# ===================================================================
# 9.  Risk Overlay
# ===================================================================

@dataclass
class RiskOverlayConfig:
    vol_target: float = 0.10
    max_leverage: float = 2.0
    max_drawdown_threshold: float = 0.10
    dd_deleverage_speed: float = 0.5
    correlation_deleverage_threshold: float = 0.7
    lookback: int = 63


class RiskOverlay:
    """Vol targeting, drawdown control, correlation-based deleverage."""

    def __init__(self, config: RiskOverlayConfig | None = None):
        self.cfg = config or RiskOverlayConfig()
        self._equity_curve: List[float] = [1.0]
        self._price_buffer: List[FloatArray] = []

    def apply(self, position: Position, prices: FloatArray, t: int) -> Position:
        self._price_buffer.append(prices.copy())
        w = position.weights.copy()

        # 1. Vol targeting
        w = self._vol_target(w)

        # 2. Leverage cap
        gross = np.abs(w).sum()
        if gross > self.cfg.max_leverage:
            w *= self.cfg.max_leverage / gross

        # 3. Drawdown control
        w = self._drawdown_control(w)

        # 4. Correlation deleverage
        w = self._correlation_deleverage(w)

        return Position(weights=w)

    def update_equity(self, ret: float) -> None:
        self._equity_curve.append(self._equity_curve[-1] * (1.0 + ret))

    def _vol_target(self, w: FloatArray) -> FloatArray:
        lb = self.cfg.lookback
        if len(self._price_buffer) < lb + 1:
            return w
        arr = np.array(self._price_buffer[-lb - 1 :])
        rets = np.diff(np.log(arr + 1e-12), axis=0)
        cov = np.cov(rets.T)
        port_var = w @ cov @ w * 252
        port_vol = np.sqrt(max(port_var, 1e-12))
        if port_vol > 1e-6:
            scale = self.cfg.vol_target / port_vol
            w *= min(scale, 3.0)
        return w

    def _drawdown_control(self, w: FloatArray) -> FloatArray:
        if len(self._equity_curve) < 2:
            return w
        eq = np.array(self._equity_curve)
        peak = eq.max()
        dd = (eq[-1] - peak) / peak
        if dd < -self.cfg.max_drawdown_threshold:
            excess = (-dd - self.cfg.max_drawdown_threshold) / self.cfg.max_drawdown_threshold
            deleverage = max(1.0 - self.cfg.dd_deleverage_speed * excess, 0.1)
            w *= deleverage
        return w

    def _correlation_deleverage(self, w: FloatArray) -> FloatArray:
        lb = self.cfg.lookback
        if len(self._price_buffer) < lb + 1:
            return w
        arr = np.array(self._price_buffer[-lb - 1 :])
        rets = np.diff(np.log(arr + 1e-12), axis=0)
        n = rets.shape[1]
        if n < 2:
            return w
        corr = np.corrcoef(rets.T)
        avg_corr = (corr.sum() - n) / (n * (n - 1))
        if avg_corr > self.cfg.correlation_deleverage_threshold:
            excess = avg_corr - self.cfg.correlation_deleverage_threshold
            scale = max(1.0 - excess * 2.0, 0.3)
            w *= scale
        return w

    @property
    def current_drawdown(self) -> float:
        eq = np.array(self._equity_curve)
        peak = eq.max()
        return float((eq[-1] - peak) / peak)


# ===================================================================
# 10. Performance Tracker
# ===================================================================

@dataclass
class PerformanceSnapshot:
    timestamp: int
    pnl: float
    cumulative_pnl: float
    weights: FloatArray
    gross_exposure: float
    net_exposure: float
    turnover: float


@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_turnover: float
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    hit_rate: float
    avg_win: float
    avg_loss: float


class PerformanceTracker:
    """Track and compute per-strategy and portfolio performance."""

    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self._returns: List[float] = []
        self._weights_history: List[FloatArray] = []
        self._snapshots: List[PerformanceSnapshot] = []
        self._strategy_returns: Dict[str, List[float]] = {}

    def record(
        self,
        t: int,
        weights: FloatArray,
        asset_returns: FloatArray,
        strategy_name: str = "portfolio",
    ) -> float:
        port_ret = float(weights @ asset_returns)
        self._returns.append(port_ret)
        self._weights_history.append(weights.copy())
        if strategy_name not in self._strategy_returns:
            self._strategy_returns[strategy_name] = []
        self._strategy_returns[strategy_name].append(port_ret)

        turnover = 0.0
        if len(self._weights_history) > 1:
            turnover = float(np.abs(weights - self._weights_history[-2]).sum())

        cum_ret = float(np.prod([1 + r for r in self._returns]) - 1.0)
        snap = PerformanceSnapshot(
            timestamp=t,
            pnl=port_ret,
            cumulative_pnl=cum_ret,
            weights=weights.copy(),
            gross_exposure=float(np.abs(weights).sum()),
            net_exposure=float(weights.sum()),
            turnover=turnover,
        )
        self._snapshots.append(snap)
        return port_ret

    def compute_metrics(self, strategy_name: str | None = None) -> PerformanceMetrics:
        if strategy_name and strategy_name in self._strategy_returns:
            rets = np.array(self._strategy_returns[strategy_name])
        else:
            rets = np.array(self._returns)
        if len(rets) < 2:
            return PerformanceMetrics(
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            )
        cum = np.cumprod(1 + rets)
        total_ret = float(cum[-1] - 1.0)
        n = len(rets)
        ann_ret = float((1 + total_ret) ** (252 / max(n, 1)) - 1)
        ann_vol = float(rets.std() * np.sqrt(252))
        sharpe = ann_ret / (ann_vol + 1e-12)
        down = rets[rets < 0]
        downside_vol = float(down.std() * np.sqrt(252)) if len(down) > 1 else 1e-6
        sortino = ann_ret / (downside_vol + 1e-12)
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / (running_max + 1e-12)
        max_dd = float(dd.min())
        calmar = ann_ret / (abs(max_dd) + 1e-12)
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        win_rate = float(len(wins) / max(n, 1))
        pf = float(wins.sum() / (abs(losses.sum()) + 1e-12)) if len(losses) > 0 else float("inf")
        turnovers = [s.turnover for s in self._snapshots]
        avg_to = float(np.mean(turnovers)) if turnovers else 0.0
        skew = float(stats.skew(rets))
        kurt = float(stats.kurtosis(rets))
        var95 = float(np.percentile(rets, 5))
        cvar95 = float(rets[rets <= var95].mean()) if (rets <= var95).any() else var95
        return PerformanceMetrics(
            total_return=total_ret,
            annualized_return=ann_ret,
            annualized_vol=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=pf,
            avg_turnover=avg_to,
            skewness=skew,
            kurtosis=kurt,
            var_95=var95,
            cvar_95=cvar95,
            hit_rate=win_rate,
            avg_win=float(wins.mean()) if len(wins) > 0 else 0.0,
            avg_loss=float(losses.mean()) if len(losses) > 0 else 0.0,
        )

    def rolling_sharpe(self, window: int = 252) -> FloatArray:
        rets = np.array(self._returns)
        n = len(rets)
        result = np.full(n, np.nan)
        for i in range(window, n):
            chunk = rets[i - window : i]
            result[i] = chunk.mean() / (chunk.std() + 1e-12) * np.sqrt(252)
        return result

    def rolling_drawdown(self) -> FloatArray:
        rets = np.array(self._returns)
        cum = np.cumprod(1 + rets)
        running_max = np.maximum.accumulate(cum)
        return (cum - running_max) / (running_max + 1e-12)

    def strategy_attribution(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for name, rets_list in self._strategy_returns.items():
            rets = np.array(rets_list)
            if len(rets) < 2:
                continue
            result[name] = {
                "total_return": float(np.prod(1 + rets) - 1),
                "annualized_vol": float(rets.std() * np.sqrt(252)),
                "sharpe": float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(252)),
                "contribution": float(rets.sum()),
            }
        return result


# ===================================================================
# 11. Transaction cost model
# ===================================================================

@dataclass
class TransactionCostConfig:
    fixed_cost_bps: float = 1.0
    spread_cost_bps: float = 2.0
    impact_coefficient: float = 0.1
    impact_exponent: float = 0.5


class TransactionCostModel:
    """Estimate trading costs from position changes."""

    def __init__(self, config: TransactionCostConfig | None = None):
        self.cfg = config or TransactionCostConfig()

    def estimate_cost(
        self,
        old_weights: FloatArray,
        new_weights: FloatArray,
        adv: FloatArray | None = None,
        notional: float = 1e8,
    ) -> float:
        trade = np.abs(new_weights - old_weights)
        dollar_trade = trade * notional
        fixed = (self.cfg.fixed_cost_bps / 1e4) * dollar_trade.sum()
        spread = (self.cfg.spread_cost_bps / 1e4) * dollar_trade.sum()
        if adv is not None:
            participation = dollar_trade / (adv + 1e-12)
            impact = self.cfg.impact_coefficient * np.sum(
                dollar_trade * participation ** self.cfg.impact_exponent
            )
        else:
            impact = 0.0
        return float(fixed + spread + impact)

    def net_return(
        self,
        gross_return: float,
        old_weights: FloatArray,
        new_weights: FloatArray,
        notional: float = 1e8,
    ) -> float:
        cost = self.estimate_cost(old_weights, new_weights, notional=notional)
        cost_pct = cost / notional
        return gross_return - cost_pct


# ===================================================================
# 12. Backtest engine
# ===================================================================

@dataclass
class BacktestConfig:
    initial_capital: float = 1e8
    transaction_cost_config: TransactionCostConfig | None = None
    risk_overlay_config: RiskOverlayConfig | None = None
    rebalance_frequency: int = 1


@dataclass
class BacktestResult:
    returns: FloatArray
    equity_curve: FloatArray
    weights_history: FloatArray
    metrics: PerformanceMetrics
    costs: FloatArray
    turnover: FloatArray


class BacktestEngine:
    """Run a strategy over historical price data."""

    def __init__(self, config: BacktestConfig | None = None):
        self.cfg = config or BacktestConfig()
        self.cost_model = TransactionCostModel(self.cfg.transaction_cost_config)
        self.risk_overlay = RiskOverlay(self.cfg.risk_overlay_config)
        self.tracker = None

    def run(
        self,
        strategy: StrategyBase,
        prices: FloatArray,
        volumes: FloatArray | None = None,
        **kwargs: Any,
    ) -> BacktestResult:
        n_steps, n_assets = prices.shape
        if volumes is None:
            volumes = np.ones_like(prices) * 1e6
        self.tracker = PerformanceTracker(n_assets)
        strategy.reset()

        returns = np.zeros(n_steps - 1)
        equity = np.ones(n_steps)
        weights_hist = np.zeros((n_steps, n_assets))
        costs = np.zeros(n_steps - 1)
        turnover = np.zeros(n_steps - 1)

        prev_weights = np.zeros(n_assets)
        for t in range(1, n_steps):
            if t % self.cfg.rebalance_frequency == 0:
                position = strategy.step(prices[t], volumes[t], t, **kwargs)
                position = self.risk_overlay.apply(position, prices[t], t)
                new_weights = position.weights
            else:
                new_weights = prev_weights

            asset_rets = prices[t] / (prices[t - 1] + 1e-12) - 1.0
            gross_ret = float(prev_weights @ asset_rets)
            cost = self.cost_model.estimate_cost(
                prev_weights, new_weights, notional=self.cfg.initial_capital
            )
            cost_pct = cost / self.cfg.initial_capital
            net_ret = gross_ret - cost_pct

            returns[t - 1] = net_ret
            equity[t] = equity[t - 1] * (1 + net_ret)
            weights_hist[t] = new_weights
            costs[t - 1] = cost_pct
            turnover[t - 1] = float(np.abs(new_weights - prev_weights).sum())

            self.tracker.record(t, new_weights, asset_rets, strategy.name)
            self.risk_overlay.update_equity(net_ret)
            prev_weights = new_weights.copy()

        metrics = self.tracker.compute_metrics()
        return BacktestResult(
            returns=returns,
            equity_curve=equity,
            weights_history=weights_hist,
            metrics=metrics,
            costs=costs,
            turnover=turnover,
        )


# ===================================================================
# 13. Portfolio optimization helpers
# ===================================================================

def minimum_variance_weights(cov: FloatArray) -> FloatArray:
    """Minimum variance portfolio."""
    n = cov.shape[0]
    inv_cov = np.linalg.inv(cov + np.eye(n) * 1e-8)
    ones = np.ones(n)
    w = inv_cov @ ones
    return w / w.sum()


def max_sharpe_weights(
    expected_returns: FloatArray, cov: FloatArray, rf: float = 0.0
) -> FloatArray:
    """Maximum Sharpe ratio portfolio (analytic)."""
    n = cov.shape[0]
    excess = expected_returns - rf
    inv_cov = np.linalg.inv(cov + np.eye(n) * 1e-8)
    w = inv_cov @ excess
    return w / (w.sum() + 1e-12)


def risk_parity_weights(cov: FloatArray, tol: float = 1e-8, max_iter: int = 1000) -> FloatArray:
    """Risk parity: each asset contributes equally to portfolio variance."""
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        sigma = np.sqrt(w @ cov @ w + 1e-12)
        mrc = cov @ w / sigma
        rc = w * mrc
        target_rc = sigma / n
        w_new = w * target_rc / (rc + 1e-12)
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            break
        w = w_new
    return w


def black_litterman_weights(
    cov: FloatArray,
    market_weights: FloatArray,
    P: FloatArray,
    Q: FloatArray,
    tau: float = 0.05,
    omega: FloatArray | None = None,
) -> FloatArray:
    """Black-Litterman posterior weights.

    Parameters
    ----------
    P : (k, n) pick matrix
    Q : (k,) view returns
    omega : (k, k) view uncertainty; if None, use tau * P @ cov @ P.T
    """
    n = cov.shape[0]
    pi = tau * cov @ market_weights  # equilibrium excess returns
    if omega is None:
        omega = np.diag(np.diag(tau * P @ cov @ P.T))
    inv_tau_cov = np.linalg.inv(tau * cov + np.eye(n) * 1e-10)
    inv_omega = np.linalg.inv(omega + np.eye(len(Q)) * 1e-10)
    M = np.linalg.inv(inv_tau_cov + P.T @ inv_omega @ P)
    mu_bl = M @ (inv_tau_cov @ pi + P.T @ inv_omega @ Q)
    w = np.linalg.inv(cov + np.eye(n) * 1e-8) @ mu_bl
    return w / (np.abs(w).sum() + 1e-12)


def hierarchical_risk_parity(cov: FloatArray) -> FloatArray:
    """Simplified HRP: recursive bisection on correlation-based clustering."""
    n = cov.shape[0]
    if n <= 1:
        return np.ones(n)
    corr = np.zeros((n, n))
    std = np.sqrt(np.diag(cov) + 1e-12)
    for i in range(n):
        for j in range(n):
            corr[i, j] = cov[i, j] / (std[i] * std[j] + 1e-12)
    dist = np.sqrt(0.5 * (1.0 - corr))
    # Simple agglomerative: split into two halves based on distance
    order = _seriation(dist)
    weights = _recursive_bisect(cov, order)
    return weights


def _seriation(dist: FloatArray) -> List[int]:
    """Simple nearest-neighbor seriation."""
    n = dist.shape[0]
    visited = [False] * n
    order = [0]
    visited[0] = True
    for _ in range(n - 1):
        last = order[-1]
        best = -1
        best_d = float("inf")
        for j in range(n):
            if not visited[j] and dist[last, j] < best_d:
                best_d = dist[last, j]
                best = j
        order.append(best)
        visited[best] = True
    return order


def _recursive_bisect(cov: FloatArray, order: List[int]) -> FloatArray:
    n = cov.shape[0]
    w = np.ones(n)
    clusters = [order]
    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]
            # Inverse variance allocation between clusters
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right + 1e-12)
            for i in left:
                w[i] *= alpha
            for i in right:
                w[i] *= (1.0 - alpha)
            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)
        clusters = new_clusters
    return w / (w.sum() + 1e-12)


def _cluster_variance(cov: FloatArray, indices: List[int]) -> float:
    sub_cov = cov[np.ix_(indices, indices)]
    n = len(indices)
    w = np.ones(n) / n
    return float(w @ sub_cov @ w)


# ===================================================================
# 14. Strategy factory
# ===================================================================

def create_strategy(name: str, n_assets: int, **kwargs: Any) -> StrategyBase:
    """Factory function to create strategies by name."""
    registry = {
        "momentum": lambda: MomentumStrategy(n_assets, MomentumConfig(**kwargs)),
        "mean_reversion": lambda: MeanReversionStrategy(n_assets, MeanReversionConfig(**kwargs)),
        "stat_arb": lambda: StatArbStrategy(n_assets, StatArbConfig(**kwargs)),
        "breakout": lambda: BreakoutStrategy(n_assets, BreakoutConfig(**kwargs)),
        "carry": lambda: CarryStrategy(n_assets, CarryConfig(**kwargs)),
        "volatility": lambda: VolatilityStrategy(n_assets, VolatilityStrategyConfig(**kwargs)),
    }
    if name not in registry:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(registry.keys())}")
    return registry[name]()


def run_all_strategies(
    prices: FloatArray,
    volumes: FloatArray | None = None,
) -> Dict[str, BacktestResult]:
    """Run all built-in strategies on the same data."""
    n_assets = prices.shape[1]
    engine = BacktestEngine()
    results = {}
    for name in ["momentum", "mean_reversion", "breakout", "carry"]:
        try:
            strat = create_strategy(name, n_assets)
            results[name] = engine.run(strat, prices, volumes)
        except Exception:
            continue
    if n_assets >= 2 and n_assets % 2 == 0:
        try:
            strat = create_strategy("stat_arb", n_assets)
            results["stat_arb"] = engine.run(strat, prices, volumes)
        except Exception:
            pass
    return results


# ===================================================================
# 15. Regime detection for blender
# ===================================================================

def detect_market_regime(
    prices: FloatArray, lookback: int = 63
) -> IntArray:
    """Classify each period into regime: 0=low_vol, 1=normal, 2=high_vol."""
    rets = np.diff(np.log(prices[:, 0] + 1e-12))
    n = len(rets)
    regimes = np.ones(n + 1, dtype=np.int64)
    for t in range(lookback, n):
        vol = rets[t - lookback : t].std() * np.sqrt(252)
        if vol < 0.12:
            regimes[t + 1] = 0
        elif vol > 0.25:
            regimes[t + 1] = 2
        else:
            regimes[t + 1] = 1
    return regimes


def detect_trend_regime(
    prices: FloatArray, short_lookback: int = 21, long_lookback: int = 126
) -> IntArray:
    """0=downtrend, 1=sideways, 2=uptrend based on moving average crossover."""
    n = prices.shape[0]
    regimes = np.ones(n, dtype=np.int64)
    for t in range(long_lookback, n):
        short_ma = prices[t - short_lookback : t, 0].mean()
        long_ma = prices[t - long_lookback : t, 0].mean()
        ratio = short_ma / (long_ma + 1e-12) - 1.0
        if ratio > 0.02:
            regimes[t] = 2
        elif ratio < -0.02:
            regimes[t] = 0
    return regimes


# ===================================================================
# __all__
# ===================================================================

__all__ = [
    "StrategyBase",
    "Signal",
    "Position",
    "StrategyState",
    "MomentumStrategy",
    "MomentumConfig",
    "MeanReversionStrategy",
    "MeanReversionConfig",
    "StatArbStrategy",
    "StatArbConfig",
    "BreakoutStrategy",
    "BreakoutConfig",
    "CarryStrategy",
    "CarryConfig",
    "VolatilityStrategy",
    "VolatilityStrategyConfig",
    "MultiStrategyBlender",
    "BlenderConfig",
    "RiskOverlay",
    "RiskOverlayConfig",
    "PerformanceTracker",
    "PerformanceMetrics",
    "TransactionCostModel",
    "TransactionCostConfig",
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "minimum_variance_weights",
    "max_sharpe_weights",
    "risk_parity_weights",
    "black_litterman_weights",
    "hierarchical_risk_parity",
    "create_strategy",
    "run_all_strategies",
    "detect_market_regime",
    "detect_trend_regime",
]
