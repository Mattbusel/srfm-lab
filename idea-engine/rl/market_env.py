"""
Custom market environment for RL agents.

OpenAI Gym-compatible interface (no gym dependency).
Supports: single-asset and multi-asset environments, discrete and continuous
action spaces, transaction costs, slippage, regime state, replay buffer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Action space definitions
# ---------------------------------------------------------------------------

class DiscreteActionSpace:
    """FLAT=0, LONG=1, SHORT=2"""
    FLAT = 0
    LONG = 1
    SHORT = 2
    n = 3

    def sample(self) -> int:
        return np.random.randint(0, self.n)

    def contains(self, action: int) -> bool:
        return 0 <= action < self.n


class ContinuousActionSpace:
    """Continuous position size in [-1, +1]. -1 = max short, +1 = max long."""
    def __init__(self, low: float = -1.0, high: float = 1.0):
        self.low = low
        self.high = high
        self.shape = (1,)

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size=(1,))

    def clip(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, self.low, self.high)

    def contains(self, action: np.ndarray) -> bool:
        return bool(np.all(action >= self.low) and np.all(action <= self.high))


# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------

@dataclass
class ObservationSpace:
    shape: Tuple[int, ...]
    low: float = -np.inf
    high: float = np.inf

    def contains(self, obs: np.ndarray) -> bool:
        return obs.shape == self.shape


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period+1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1 + rs)


def _compute_macd(prices: np.ndarray, fast: int = 12,
                  slow: int = 26) -> Tuple[float, float]:
    """Returns (macd_line, signal). Simplified via SMA."""
    if len(prices) < slow:
        return 0.0, 0.0
    ema_fast = prices[-fast:].mean()
    ema_slow = prices[-slow:].mean()
    macd = ema_fast - ema_slow
    # Signal: 9-period EMA of macd (simplified as SMA here)
    signal = macd  # single observation; full impl needs history
    return macd, signal


def _compute_bollinger(prices: np.ndarray,
                       period: int = 20) -> Tuple[float, float, float]:
    """Returns (upper, mid, lower) bands."""
    if len(prices) < period:
        mu = float(prices.mean())
        return mu, mu, mu
    window = prices[-period:]
    mu = float(window.mean())
    std = float(window.std())
    return mu + 2 * std, mu, mu - 2 * std


def _compute_atr(highs: np.ndarray, lows: np.ndarray,
                 closes: np.ndarray, period: int = 14) -> float:
    """Average True Range."""
    n = min(len(highs), len(lows), len(closes))
    if n < 2:
        return 0.0
    h, l, c = highs[-n:], lows[-n:], closes[-n:]
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    return float(tr[-period:].mean()) if len(tr) >= period else float(tr.mean())


def _detect_regime(returns: np.ndarray, window: int = 20) -> int:
    """Simple regime: 0=trending_up, 1=trending_down, 2=ranging."""
    if len(returns) < window:
        return 2
    r = returns[-window:]
    mean_r = float(r.mean())
    vol_r = float(r.std()) + 1e-8
    z = mean_r / vol_r * math.sqrt(window)
    if z > 1.5:
        return 0  # trending up
    elif z < -1.5:
        return 1  # trending down
    return 2  # ranging


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Experience replay buffer for off-policy RL algorithms.
    Stores (state, action, reward, next_state, done) tuples.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int = 1):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, state: np.ndarray, action: Any, reward: float,
             next_state: np.ndarray, done: bool):
        self.states[self.ptr] = state
        self.actions[self.ptr] = np.atleast_1d(action).astype(np.float32)
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.choice(self.size, size=min(batch_size, self.size), replace=False)
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
        }

    def __len__(self) -> int:
        return self.size

    def is_ready(self, min_size: int) -> bool:
        return self.size >= min_size


# ---------------------------------------------------------------------------
# Single-Asset Market Environment
# ---------------------------------------------------------------------------

class MarketEnv:
    """
    Single-asset market environment for RL trading agents.
    Gym-compatible: reset() → obs, step(action) → (obs, reward, done, info).

    Parameters
    ----------
    prices       : np.ndarray, shape (T,)  — close prices
    highs        : optional, shape (T,)
    lows         : optional, shape (T,)
    volumes      : optional, shape (T,)
    window       : int — lookback window for feature construction
    episode_len  : int — steps per episode (None = full series)
    tc_bps       : float — transaction costs in basis points
    slippage_vol_mult : float — slippage as multiple of realized vol
    discrete     : bool — True = discrete actions {FLAT, LONG, SHORT}
    max_position : float — max absolute position (units of 1 = fully invested)
    reward_scaling : float — scale rewards for gradient stability
    sharpe_window  : int — rolling window for Sharpe-based reward
    drawdown_penalty : float — penalty coefficient for drawdown
    """

    def __init__(
        self,
        prices: np.ndarray,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
        volumes: Optional[np.ndarray] = None,
        window: int = 30,
        episode_len: Optional[int] = None,
        tc_bps: float = 5.0,
        slippage_vol_mult: float = 0.1,
        discrete: bool = True,
        max_position: float = 1.0,
        reward_scaling: float = 1.0,
        sharpe_window: int = 60,
        drawdown_penalty: float = 0.1,
    ):
        self.prices = np.asarray(prices, dtype=np.float64)
        self.highs = np.asarray(highs if highs is not None else prices * 1.001)
        self.lows = np.asarray(lows if lows is not None else prices * 0.999)
        self.volumes = np.asarray(volumes if volumes is not None else np.ones(len(prices)))
        self.T = len(prices)
        self.window = window
        self.episode_len = episode_len or (self.T - window - 1)
        self.tc_bps = tc_bps / 10000.0
        self.slippage_vol_mult = slippage_vol_mult
        self.discrete = discrete
        self.max_position = max_position
        self.reward_scaling = reward_scaling
        self.sharpe_window = sharpe_window
        self.drawdown_penalty = drawdown_penalty

        # Precompute log returns
        self.log_returns = np.diff(np.log(self.prices), prepend=np.log(self.prices[0]))

        # Action / observation spaces
        self.action_space = DiscreteActionSpace() if discrete else ContinuousActionSpace()
        n_features = self._obs_dim()
        self.observation_space = ObservationSpace(shape=(n_features,))

        # Episode state
        self._t: int = window
        self._start: int = window
        self._position: float = 0.0
        self._equity: float = 1.0
        self._peak_equity: float = 1.0
        self._step_count: int = 0
        self._return_history: List[float] = []
        self._trade_log: List[Dict] = []
        self._n_trades: int = 0

    def _obs_dim(self) -> int:
        # Features: window returns (window), RSI, MACD, BollBand z-score,
        # ATR normalized, position, pnl, regime (3 one-hot) = window + 8
        return self.window + 8

    def _get_obs(self) -> np.ndarray:
        t = self._t
        w = self.window
        p = self.prices[t - w: t]
        rets = self.log_returns[t - w: t]

        # Normalize returns
        ret_std = float(rets.std()) + 1e-8
        rets_norm = rets / ret_std

        # Technical features
        rsi = (_compute_rsi(p) - 50.0) / 50.0
        macd, _ = _compute_macd(p)
        macd_norm = macd / (float(p[-1]) + 1e-8)
        upper, mid, lower = _compute_bollinger(p)
        boll_z = (float(p[-1]) - mid) / (upper - lower + 1e-8)
        atr = _compute_atr(self.highs[t-w:t], self.lows[t-w:t], p)
        atr_norm = atr / (float(p[-1]) + 1e-8)

        # Portfolio features
        pos_norm = self._position / self.max_position
        pnl_norm = (self._equity - 1.0)

        # Regime one-hot
        regime = _detect_regime(rets)
        regime_oh = np.zeros(3)
        regime_oh[regime] = 1.0

        obs = np.concatenate([
            rets_norm,
            [rsi, macd_norm, boll_z, atr_norm, pos_norm, pnl_norm - float(regime_oh.sum() == 0)],
            regime_oh,
        ])
        # Drop last regime_oh from extra scalar (clean up count)
        # Recount: window + rsi + macd_norm + boll_z + atr_norm + pos + pnl + 3 regime = window+8
        return obs[:self._obs_dim()].astype(np.float32)

    def _action_to_position(self, action: Any) -> float:
        if self.discrete:
            mapping = {
                DiscreteActionSpace.FLAT: 0.0,
                DiscreteActionSpace.LONG: self.max_position,
                DiscreteActionSpace.SHORT: -self.max_position,
            }
            return mapping.get(int(action), 0.0)
        else:
            return float(np.clip(action, -self.max_position, self.max_position))

    def _compute_slippage(self) -> float:
        t = self._t
        w = min(self.window, t)
        vol = float(self.log_returns[t-w:t].std()) + 1e-8
        return vol * self.slippage_vol_mult

    def _compute_reward(self, raw_return: float, position_change: float) -> float:
        """Sharpe-based reward with drawdown penalty."""
        self._return_history.append(raw_return)
        if len(self._return_history) > self.sharpe_window:
            self._return_history.pop(0)

        hist = np.array(self._return_history)
        if len(hist) < 2:
            base_reward = raw_return
        else:
            mu = hist.mean()
            sigma = hist.std() + 1e-8
            # Incremental Sharpe contribution
            base_reward = mu / sigma * math.sqrt(252 / max(len(hist), 1))

        # Transaction cost
        tc = abs(position_change) * self.tc_bps
        slippage = abs(position_change) * self._compute_slippage()

        # Drawdown penalty
        dd = (self._peak_equity - self._equity) / (self._peak_equity + 1e-8)
        dd_penalty = self.drawdown_penalty * max(dd, 0.0)

        reward = (base_reward - tc - slippage - dd_penalty) * self.reward_scaling
        return float(reward)

    def reset(self, start: Optional[int] = None) -> np.ndarray:
        """Reset environment. Returns initial observation."""
        max_start = self.T - self.episode_len - 1
        if start is not None:
            self._start = max(self.window, min(start, max_start))
        else:
            self._start = np.random.randint(self.window, max(self.window + 1, max_start))
        self._t = self._start
        self._position = 0.0
        self._equity = 1.0
        self._peak_equity = 1.0
        self._step_count = 0
        self._return_history.clear()
        self._trade_log.clear()
        self._n_trades = 0
        return self._get_obs()

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action, advance environment by one step.
        Returns (obs, reward, done, info).
        """
        target_pos = self._action_to_position(action)
        position_change = target_pos - self._position

        # Apply slippage to entry/exit
        slippage = self._compute_slippage() * abs(position_change)
        self._position = target_pos
        if abs(position_change) > 1e-6:
            self._n_trades += 1

        # Advance time
        self._t += 1
        self._step_count += 1

        # Price return
        price_return = float(self.log_returns[self._t])
        portfolio_return = self._position * price_return - slippage - abs(position_change) * self.tc_bps

        self._equity *= math.exp(portfolio_return)
        self._peak_equity = max(self._peak_equity, self._equity)

        reward = self._compute_reward(portfolio_return, position_change)

        done = (self._step_count >= self.episode_len or self._t >= self.T - 1)
        obs = self._get_obs()

        info = self._build_info()
        return obs, reward, done, info

    def _build_info(self) -> Dict:
        hist = np.array(self._return_history) if self._return_history else np.array([0.0])
        mu = float(hist.mean())
        sigma = float(hist.std()) + 1e-8
        sharpe = mu / sigma * math.sqrt(252) if len(hist) > 1 else 0.0
        drawdown = float((self._peak_equity - self._equity) / (self._peak_equity + 1e-8))
        return {
            "equity": self._equity,
            "position": self._position,
            "pnl": self._equity - 1.0,
            "sharpe": sharpe,
            "max_drawdown": drawdown,
            "n_trades": self._n_trades,
            "step": self._step_count,
            "t": self._t,
        }

    def render(self) -> str:
        info = self._build_info()
        return (
            f"t={info['t']} | equity={info['equity']:.4f} | "
            f"pos={info['position']:+.3f} | "
            f"pnl={info['pnl']:+.4f} | "
            f"sharpe={info['sharpe']:.3f} | "
            f"dd={info['max_drawdown']:.4f} | "
            f"trades={info['n_trades']}"
        )


# ---------------------------------------------------------------------------
# Multi-Asset Market Environment
# ---------------------------------------------------------------------------

class MultiAssetEnv:
    """
    Multi-asset environment where the agent allocates portfolio weights
    across N assets simultaneously.

    Actions: continuous weight vector in simplex (long-only) or [-1,1]^N (long-short).
    """

    def __init__(
        self,
        prices: np.ndarray,           # (T, N)
        window: int = 30,
        episode_len: Optional[int] = None,
        tc_bps: float = 5.0,
        slippage_vol_mult: float = 0.1,
        long_only: bool = True,
        max_leverage: float = 1.0,
        reward_scaling: float = 1.0,
        sharpe_window: int = 60,
        drawdown_penalty: float = 0.1,
    ):
        self.prices = np.asarray(prices, dtype=np.float64)
        assert self.prices.ndim == 2
        self.T, self.N = self.prices.shape
        self.window = window
        self.episode_len = episode_len or (self.T - window - 1)
        self.tc_bps = tc_bps / 10000.0
        self.slippage_vol_mult = slippage_vol_mult
        self.long_only = long_only
        self.max_leverage = max_leverage
        self.reward_scaling = reward_scaling
        self.sharpe_window = sharpe_window
        self.drawdown_penalty = drawdown_penalty

        self.log_returns = np.diff(np.log(self.prices), axis=0, prepend=np.log(self.prices[:1]))

        low = 0.0 if long_only else -1.0
        self.action_space = ContinuousActionSpace(low=low, high=1.0)
        n_feat = self.N * window + self.N + 3  # returns + current weights + portfolio features
        self.observation_space = ObservationSpace(shape=(n_feat,))

        self._t: int = window
        self._start: int = window
        self._weights = np.ones(self.N) / self.N
        self._equity: float = 1.0
        self._peak_equity: float = 1.0
        self._step_count: int = 0
        self._return_history: List[float] = []
        self._n_trades: int = 0

    def _get_obs(self) -> np.ndarray:
        t = self._t
        w = self.window
        rets = self.log_returns[t-w:t, :]  # (w, N)
        # Normalize each asset's returns
        std = rets.std(axis=0) + 1e-8
        rets_norm = (rets / std).ravel()
        equity_feat = np.array([self._equity - 1.0,
                                 (self._peak_equity - self._equity) / (self._peak_equity + 1e-8),
                                 float(self._n_trades)])
        return np.concatenate([rets_norm, self._weights, equity_feat]).astype(np.float32)

    def _normalize_weights(self, action: np.ndarray) -> np.ndarray:
        """Project action to valid portfolio weights."""
        action = np.asarray(action, dtype=float).ravel()[:self.N]
        if self.long_only:
            action = np.maximum(action, 0.0)
            s = action.sum()
            if s < 1e-8:
                return np.ones(self.N) / self.N
            return action / s * self.max_leverage
        else:
            # Allow long-short; normalize by L1 norm, scale by max_leverage
            s = np.sum(np.abs(action)) + 1e-8
            return action / s * self.max_leverage

    def reset(self, start: Optional[int] = None) -> np.ndarray:
        max_start = self.T - self.episode_len - 1
        if start is not None:
            self._start = max(self.window, min(start, max_start))
        else:
            self._start = np.random.randint(self.window, max(self.window + 1, max_start))
        self._t = self._start
        self._weights = np.ones(self.N) / self.N
        self._equity = 1.0
        self._peak_equity = 1.0
        self._step_count = 0
        self._return_history.clear()
        self._n_trades = 0
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        new_weights = self._normalize_weights(action)
        weight_change = np.sum(np.abs(new_weights - self._weights))
        tc = weight_change * self.tc_bps

        # Compute realized vol for slippage
        w = self.window
        t = self._t
        vol = float(self.log_returns[t-w:t, :].std())
        slippage = weight_change * self.slippage_vol_mult * vol

        if weight_change > 1e-4:
            self._n_trades += 1
        self._weights = new_weights

        self._t += 1
        self._step_count += 1

        asset_returns = self.log_returns[self._t]
        port_return = float(self._weights @ asset_returns) - tc - slippage
        self._equity *= math.exp(port_return)
        self._peak_equity = max(self._peak_equity, self._equity)

        self._return_history.append(port_return)
        if len(self._return_history) > self.sharpe_window:
            self._return_history.pop(0)

        hist = np.array(self._return_history)
        if len(hist) < 2:
            reward = port_return
        else:
            mu = hist.mean()
            sigma = hist.std() + 1e-8
            reward = mu / sigma * math.sqrt(252)

        dd = (self._peak_equity - self._equity) / (self._peak_equity + 1e-8)
        reward -= self.drawdown_penalty * max(dd, 0.0)
        reward *= self.reward_scaling

        done = self._step_count >= self.episode_len or self._t >= self.T - 1
        obs = self._get_obs()
        info = {
            "equity": self._equity,
            "pnl": self._equity - 1.0,
            "weights": self._weights.copy(),
            "max_drawdown": dd,
            "n_trades": self._n_trades,
            "sharpe": float(hist.mean() / (hist.std() + 1e-8) * math.sqrt(252)) if len(hist) > 1 else 0.0,
        }
        return obs, reward, done, info

    def render(self) -> str:
        dd = (self._peak_equity - self._equity) / (self._peak_equity + 1e-8)
        return (
            f"t={self._t} | equity={self._equity:.4f} | "
            f"weights={np.round(self._weights, 3).tolist()} | "
            f"dd={dd:.4f} | trades={self._n_trades}"
        )
