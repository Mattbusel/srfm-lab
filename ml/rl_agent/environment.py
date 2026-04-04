"""
TradingEnv: OpenAI Gym-compatible reinforcement learning environment for multi-asset trading.

Supports continuous position sizing, Sharpe-adjusted rewards, drawdown penalties,
and rich state spaces including OHLCV data, buy-and-hold (BH) features, positions, and PnL.
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    warnings.warn("gym not installed, using stub spaces")
    GYM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Instrument:
    """Single tradeable instrument."""
    symbol: str
    tick_size: float = 0.01
    lot_size: float = 1.0
    margin_rate: float = 0.1          # fraction of notional
    transaction_cost: float = 0.0005  # 5 bps per trade
    slippage: float = 0.0002          # 2 bps market impact
    max_position: float = 1.0         # normalized (-1 to +1)


@dataclass
class TradingConfig:
    """Environment configuration."""
    initial_capital: float = 100_000.0
    max_episode_steps: int = 252          # ~1 trading year
    window_size: int = 60                 # lookback for OHLCV features
    reward_scaling: float = 1.0
    sharpe_annualize: float = 252.0
    drawdown_penalty_coef: float = 0.5
    risk_free_rate: float = 0.02 / 252    # daily risk-free
    entropy_bonus: float = 0.001
    use_bh_features: bool = True
    use_macro_features: bool = False
    position_penalty: float = 0.0001     # cost of holding positions (carry)
    reward_type: str = "sharpe"          # "sharpe" | "pnl" | "calmar"
    max_drawdown_limit: float = 0.25     # terminate if exceeds this
    curriculum_level: int = 0            # 0=easy, 1=medium, 2=hard


@dataclass
class EpisodeState:
    """Mutable episode state."""
    step: int = 0
    portfolio_value: float = 100_000.0
    cash: float = 100_000.0
    positions: np.ndarray = field(default_factory=lambda: np.zeros(1))
    avg_entry_prices: np.ndarray = field(default_factory=lambda: np.zeros(1))
    pnl_history: List[float] = field(default_factory=list)
    return_history: List[float] = field(default_factory=list)
    peak_value: float = 100_000.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    episode_sharpe: float = 0.0


# ---------------------------------------------------------------------------
# Feature engineering helpers (used inline in env)
# ---------------------------------------------------------------------------

def _compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Compute RSI for the last `period` returns."""
    if len(prices) < period + 1:
        return 0.5
    deltas = np.diff(prices[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean() + 1e-8
    avg_loss = losses.mean() + 1e-8
    rs = avg_gain / avg_loss
    return float(1.0 - 1.0 / (1.0 + rs))


def _compute_bollinger_bands(prices: np.ndarray, period: int = 20) -> Tuple[float, float, float]:
    """Return (upper_z, mid_z, lower_z) normalized by std."""
    if len(prices) < period:
        return 0.0, 0.0, 0.0
    window = prices[-period:]
    mu = window.mean()
    sigma = window.std() + 1e-8
    upper = (mu + 2 * sigma - prices[-1]) / sigma
    lower = (mu - 2 * sigma - prices[-1]) / sigma
    mid = (mu - prices[-1]) / sigma
    return float(upper), float(mid), float(lower)


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Average True Range, normalized by close."""
    if len(close) < period + 1:
        return 0.0
    tr_list = []
    for i in range(-period, 0):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr_list.append(max(hl, hc, lc))
    atr = np.mean(tr_list)
    return float(atr / (close[-1] + 1e-8))


def _compute_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
    """Return (MACD line, signal line) normalized by price."""
    if len(prices) < slow + signal:
        return 0.0, 0.0
    ema_fast = pd.Series(prices).ewm(span=fast).mean().values
    ema_slow = pd.Series(prices).ewm(span=slow).mean().values
    macd_line = ema_fast - ema_slow
    sig_line = pd.Series(macd_line).ewm(span=signal).mean().values
    norm = prices[-1] + 1e-8
    return float(macd_line[-1] / norm), float(sig_line[-1] / norm)


def _rolling_zscore(series: np.ndarray, window: int = 20) -> float:
    """Z-score of last value relative to rolling window."""
    if len(series) < window:
        return 0.0
    w = series[-window:]
    mu = w.mean()
    sigma = w.std() + 1e-8
    return float((series[-1] - mu) / sigma)


# ---------------------------------------------------------------------------
# Buy-and-Hold baseline state
# ---------------------------------------------------------------------------

class BHBaseline:
    """
    Tracks a simple buy-and-hold (BH) baseline for each instrument.
    The RL agent receives features comparing its performance to the BH baseline.
    """

    def __init__(self, n_instruments: int, initial_capital: float):
        self.n_instruments = n_instruments
        self.initial_capital = initial_capital
        self.bh_value: float = initial_capital
        self.bh_positions: np.ndarray = np.ones(n_instruments) / n_instruments
        self.bh_entry_prices: Optional[np.ndarray] = None
        self.bh_returns: List[float] = []

    def initialize(self, current_prices: np.ndarray) -> None:
        """Set entry prices on episode start."""
        self.bh_entry_prices = current_prices.copy()
        self.bh_returns = []
        self.bh_value = self.initial_capital

    def step(self, current_prices: np.ndarray) -> float:
        """Update BH value and return daily BH return."""
        if self.bh_entry_prices is None:
            return 0.0
        weights = self.bh_positions / (self.bh_positions.sum() + 1e-8)
        price_returns = (current_prices - self.bh_entry_prices) / (self.bh_entry_prices + 1e-8)
        portfolio_return = float(np.dot(weights, price_returns))
        self.bh_returns.append(portfolio_return)
        self.bh_value = self.initial_capital * (1.0 + portfolio_return)
        return portfolio_return

    def get_features(self, agent_value: float, agent_return_series: List[float]) -> np.ndarray:
        """
        Return features comparing agent to BH:
        [relative_value, bh_sharpe, agent_sharpe, excess_return, tracking_error]
        """
        relative_value = float(agent_value / (self.bh_value + 1e-8) - 1.0)

        def _sharpe(returns: List[float]) -> float:
            if len(returns) < 2:
                return 0.0
            r = np.array(returns)
            return float(r.mean() / (r.std() + 1e-8) * np.sqrt(252))

        bh_sharpe = _sharpe(self.bh_returns[-60:])
        agent_sharpe = _sharpe(agent_return_series[-60:])

        if self.bh_returns and agent_return_series:
            min_len = min(len(self.bh_returns), len(agent_return_series))
            excess = np.array(agent_return_series[-min_len:]) - np.array(self.bh_returns[-min_len:])
            excess_return = float(excess.mean())
            tracking_error = float(excess.std() + 1e-8)
        else:
            excess_return = 0.0
            tracking_error = 1.0

        return np.array([relative_value, bh_sharpe, agent_sharpe, excess_return, tracking_error],
                        dtype=np.float32)


# ---------------------------------------------------------------------------
# Observation space builder
# ---------------------------------------------------------------------------

class ObservationBuilder:
    """Constructs the observation vector for each step."""

    # Feature dimensions (per instrument)
    OHLCV_FEATURES = 20       # normalized OHLCV + returns + stats
    TECHNICAL_FEATURES = 12   # RSI, BB, ATR, MACD, Z-scores, etc.
    POSITION_FEATURES = 3     # position, unrealized_pnl, hold_duration
    BH_FEATURES = 5           # vs-BH comparison
    GLOBAL_FEATURES = 8       # time, capital, drawdown, etc.

    def __init__(self, config: TradingConfig, instruments: List[Instrument]):
        self.config = config
        self.instruments = instruments
        self.n = len(instruments)
        self.window = config.window_size

        per_instrument = (
            self.OHLCV_FEATURES
            + self.TECHNICAL_FEATURES
            + self.POSITION_FEATURES
        )
        if config.use_bh_features:
            per_instrument += self.BH_FEATURES

        self.obs_dim = per_instrument * self.n + self.GLOBAL_FEATURES

    def build(
        self,
        ohlcv_history: np.ndarray,   # (window, n_instruments, 5)
        state: EpisodeState,
        current_prices: np.ndarray,
        bh_baseline: Optional[BHBaseline],
        episode_progress: float,
    ) -> np.ndarray:
        """Build observation vector."""
        obs_parts = []

        for i in range(self.n):
            history = ohlcv_history[:, i, :]  # (window, 5)
            opens   = history[:, 0]
            highs   = history[:, 1]
            lows    = history[:, 2]
            closes  = history[:, 3]
            volumes = history[:, 4]

            # --- OHLCV features ---
            log_returns = np.diff(np.log(closes + 1e-8))
            log_ret_mean = log_returns.mean() if len(log_returns) > 0 else 0.0
            log_ret_std  = log_returns.std() + 1e-8 if len(log_returns) > 0 else 1.0

            vol_norm = (volumes - volumes.mean()) / (volumes.std() + 1e-8)
            recent_vol = float(vol_norm[-1]) if len(vol_norm) > 0 else 0.0

            close_norm = (closes - closes.mean()) / (closes.std() + 1e-8)
            high_norm  = (highs  - closes.mean()) / (closes.std() + 1e-8)
            low_norm   = (lows   - closes.mean()) / (closes.std() + 1e-8)

            ohlcv_feat = np.array([
                float(close_norm[-1]) if len(close_norm) > 0 else 0.0,
                float(high_norm[-1])  if len(high_norm)  > 0 else 0.0,
                float(low_norm[-1])   if len(low_norm)   > 0 else 0.0,
                recent_vol,
                log_ret_mean / (log_ret_std + 1e-8),
                float(log_returns[-1]) / (log_ret_std + 1e-8) if len(log_returns) > 0 else 0.0,
                float(log_returns[-5:].mean()) / (log_ret_std + 1e-8) if len(log_returns) >= 5 else 0.0,
                float(log_returns[-20:].mean()) / (log_ret_std + 1e-8) if len(log_returns) >= 20 else 0.0,
                float(log_returns[-5:].std()) / (log_ret_std + 1e-8) if len(log_returns) >= 5 else 0.0,
                float(log_returns[-20:].std()) / (log_ret_std + 1e-8) if len(log_returns) >= 20 else 0.0,
                float((closes[-1] - closes[-2]) / (closes[-2] + 1e-8)) if len(closes) >= 2 else 0.0,
                float((closes[-1] - closes[-5]) / (closes[-5] + 1e-8)) if len(closes) >= 5 else 0.0,
                float((closes[-1] - closes[-20]) / (closes[-20] + 1e-8)) if len(closes) >= 20 else 0.0,
                float((closes[-1] - closes[-60]) / (closes[-60] + 1e-8)) if len(closes) >= 60 else 0.0,
                float((highs[-1] - lows[-1]) / (closes[-1] + 1e-8)),         # intraday range
                float((highs[-5:].max() - lows[-5:].min()) / (closes[-1] + 1e-8)) if len(highs) >= 5 else 0.0,
                float(np.percentile(closes, 80) - np.percentile(closes, 20)) / (closes.mean() + 1e-8),
                float(log_returns[-10:].sum()) / (log_ret_std + 1e-8) if len(log_returns) >= 10 else 0.0,
                float(np.sign(log_returns[-5:]).mean()) if len(log_returns) >= 5 else 0.0,
                float(((closes[-1] - closes.min()) / (closes.max() - closes.min() + 1e-8)))
            ], dtype=np.float32)

            # --- Technical features ---
            rsi = _compute_rsi(closes)
            bb_upper, bb_mid, bb_lower = _compute_bollinger_bands(closes)
            atr = _compute_atr(highs, lows, closes)
            macd_line, macd_sig = _compute_macd(closes)
            zscore_5  = _rolling_zscore(closes, 5)
            zscore_20 = _rolling_zscore(closes, 20)
            zscore_60 = _rolling_zscore(closes, 60)
            vol_zscore = _rolling_zscore(volumes, 20)

            # Momentum features
            mom_1  = float((closes[-1] - closes[-2]) / (closes[-2] + 1e-8)) if len(closes) >= 2 else 0.0
            mom_5  = float((closes[-1] - closes[-6]) / (closes[-6] + 1e-8)) if len(closes) >= 6 else 0.0
            mom_20 = float((closes[-1] - closes[-21]) / (closes[-21] + 1e-8)) if len(closes) >= 21 else 0.0

            tech_feat = np.array([
                rsi, bb_upper, bb_mid, bb_lower, atr,
                macd_line, macd_sig,
                zscore_5, zscore_20, zscore_60,
                vol_zscore, mom_20,
            ], dtype=np.float32)

            # --- Position features ---
            pos = float(state.positions[i])
            entry = float(state.avg_entry_prices[i])
            current_price = float(current_prices[i])
            unrealized_pnl = pos * (current_price - entry) / (entry + 1e-8) if abs(entry) > 1e-8 else 0.0
            pos_feat = np.array([pos, unrealized_pnl, float(state.step) / max(self.config.max_episode_steps, 1)],
                                dtype=np.float32)

            # --- BH features ---
            instrument_features = [ohlcv_feat, tech_feat, pos_feat]
            if self.config.use_bh_features and bh_baseline is not None:
                bh_feat = bh_baseline.get_features(state.portfolio_value, state.return_history)
                instrument_features.append(bh_feat)

            obs_parts.extend(instrument_features)

        # --- Global features ---
        portfolio_norm = float(state.portfolio_value / self.config.initial_capital - 1.0)
        cash_ratio = float(state.cash / (state.portfolio_value + 1e-8))
        drawdown = float(state.max_drawdown)
        episode_prog = float(episode_progress)
        total_pos = float(np.abs(state.positions).sum() / self.n)

        recent_returns = state.return_history[-20:]
        sharpe_est = float(np.mean(recent_returns) / (np.std(recent_returns) + 1e-8) * np.sqrt(252)) \
            if len(recent_returns) >= 2 else 0.0

        trade_rate = float(state.total_trades / max(state.step, 1))

        win_rate = float(state.winning_trades / max(state.total_trades, 1))

        global_feat = np.array([
            portfolio_norm,
            cash_ratio,
            drawdown,
            episode_prog,
            total_pos,
            sharpe_est,
            trade_rate,
            win_rate,
        ], dtype=np.float32)

        obs_parts.append(global_feat)

        obs = np.concatenate([p.flatten() for p in obs_parts], axis=0)
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        obs = np.clip(obs, -10.0, 10.0)
        return obs.astype(np.float32)


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

class RewardComputer:
    """Computes the reward signal for the RL agent."""

    def __init__(self, config: TradingConfig):
        self.config = config
        self._return_buffer: deque = deque(maxlen=60)

    def reset(self) -> None:
        self._return_buffer.clear()

    def compute(
        self,
        step_return: float,
        state: EpisodeState,
        transaction_costs: float,
        position_changes: np.ndarray,
        drawdown: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute reward and return component breakdown."""
        self._return_buffer.append(step_return)
        returns_arr = np.array(self._return_buffer)

        reward = 0.0
        info: Dict[str, float] = {}

        if self.config.reward_type == "sharpe":
            if len(returns_arr) >= 2:
                excess = returns_arr - self.config.risk_free_rate
                sharpe_step = float(excess.mean() / (returns_arr.std() + 1e-8) * np.sqrt(self.config.sharpe_annualize))
            else:
                sharpe_step = float(step_return - self.config.risk_free_rate) * np.sqrt(self.config.sharpe_annualize)
            reward += sharpe_step * self.config.reward_scaling
            info["sharpe_reward"] = sharpe_step

        elif self.config.reward_type == "pnl":
            reward += step_return * self.config.reward_scaling
            info["pnl_reward"] = step_return

        elif self.config.reward_type == "calmar":
            ann_return = step_return * 252
            calmar = ann_return / (drawdown + 1e-4)
            reward += float(np.clip(calmar, -10.0, 10.0)) * self.config.reward_scaling
            info["calmar_reward"] = float(calmar)

        # Drawdown penalty
        dd_penalty = -self.config.drawdown_penalty_coef * (drawdown ** 2)
        reward += dd_penalty
        info["drawdown_penalty"] = dd_penalty

        # Transaction cost penalty
        tc_penalty = -float(transaction_costs) * 100
        reward += tc_penalty
        info["tc_penalty"] = tc_penalty

        # Position holding cost
        pos_penalty = -self.config.position_penalty * float(np.abs(position_changes).sum())
        reward += pos_penalty
        info["position_penalty"] = pos_penalty

        info["total_reward"] = reward
        info["step_return"] = step_return

        return float(reward), info


# ---------------------------------------------------------------------------
# Main TradingEnv
# ---------------------------------------------------------------------------

class TradingEnv:
    """
    OpenAI Gym-compatible multi-asset trading environment.

    Observation space: concatenated OHLCV, technical, BH, and position features.
    Action space: continuous position sizes in [-1, +1] per instrument.
    Reward: Sharpe-adjusted return minus drawdown penalty minus transaction costs.
    """

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        instruments: Optional[List[Instrument]] = None,
        config: Optional[TradingConfig] = None,
    ):
        self.config = config or TradingConfig()

        # Normalize data to dict[symbol -> DataFrame with OHLCV]
        if isinstance(data, pd.DataFrame):
            sym = "ASSET"
            self._data: Dict[str, pd.DataFrame] = {sym: data}
        else:
            self._data = data

        if instruments is None:
            self.instruments = [Instrument(symbol=s) for s in self._data]
        else:
            self.instruments = instruments

        self.n_instruments = len(self.instruments)
        self._validate_data()

        # Align all data to common index
        self._aligned_data = self._align_data()  # shape (T, n, 5)
        self.T = len(self._aligned_data)

        # Build obs/action spaces
        self.obs_builder = ObservationBuilder(self.config, self.instruments)
        self.obs_dim = self.obs_builder.obs_dim
        self.act_dim = self.n_instruments

        if GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=-10.0, high=10.0, shape=(self.obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32
            )
        else:
            self.observation_space = None
            self.action_space = None

        # State
        self._state: Optional[EpisodeState] = None
        self._current_idx: int = 0
        self._episode_start_idx: int = 0
        self._bh_baseline = BHBaseline(self.n_instruments, self.config.initial_capital)
        self._reward_computer = RewardComputer(self.config)
        self._ohlcv_window: deque = deque(maxlen=self.config.window_size)

        # Curriculum: episode start ranges
        self._curriculum_ranges = {
            0: (0.0, 0.5),    # easy: first half (lower volatility expected)
            1: (0.0, 0.75),   # medium: first 75%
            2: (0.0, 1.0),    # hard: full dataset
        }

        self.np_random = np.random.default_rng(42)
        self._info_history: List[Dict] = []

    def _validate_data(self) -> None:
        for sym, df in self._data.items():
            required = ["open", "high", "low", "close", "volume"]
            cols = [c.lower() for c in df.columns]
            for req in required:
                if req not in cols:
                    raise ValueError(f"Instrument {sym} missing column: {req}")

    def _align_data(self) -> np.ndarray:
        """Align all instruments to a common date index, return (T, n, 5)."""
        dfs = []
        for instr in self.instruments:
            df = self._data[instr.symbol].copy()
            df.columns = [c.lower() for c in df.columns]
            arr = df[["open", "high", "low", "close", "volume"]].values.astype(np.float32)
            dfs.append(arr)

        min_len = min(len(d) for d in dfs)
        aligned = np.stack([d[-min_len:] for d in dfs], axis=1)  # (T, n, 5)
        return aligned

    def seed(self, seed: Optional[int] = None) -> List[int]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        return [seed or 0]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.seed(seed)

        # Pick episode start based on curriculum
        level = self.config.curriculum_level
        low_frac, high_frac = self._curriculum_ranges.get(level, (0.0, 1.0))
        low_idx = int(low_frac * self.T)
        high_idx = int(high_frac * self.T) - self.config.max_episode_steps - self.config.window_size
        high_idx = max(high_idx, low_idx + 1)

        start = int(self.np_random.integers(low_idx, high_idx))
        self._episode_start_idx = start
        self._current_idx = start

        # Initialize state
        self._state = EpisodeState(
            step=0,
            portfolio_value=self.config.initial_capital,
            cash=self.config.initial_capital,
            positions=np.zeros(self.n_instruments, dtype=np.float32),
            avg_entry_prices=np.zeros(self.n_instruments, dtype=np.float32),
            peak_value=self.config.initial_capital,
        )

        # Fill initial window
        self._ohlcv_window.clear()
        window_start = max(0, start - self.config.window_size)
        for idx in range(window_start, start):
            self._ohlcv_window.append(self._aligned_data[idx])

        # Pad if needed
        while len(self._ohlcv_window) < self.config.window_size:
            self._ohlcv_window.appendleft(self._aligned_data[window_start])

        # Initialize BH baseline
        current_prices = self._get_current_prices()
        self._bh_baseline.initialize(current_prices)
        self._reward_computer.reset()
        self._info_history.clear()

        obs = self._build_observation()
        return obs, {}

    def _get_current_prices(self) -> np.ndarray:
        """Get closing prices at current index."""
        return self._aligned_data[self._current_idx, :, 3].copy()

    def _get_ohlcv_array(self) -> np.ndarray:
        """Get (window, n, 5) array from deque."""
        return np.array(list(self._ohlcv_window), dtype=np.float32)

    def _build_observation(self) -> np.ndarray:
        ohlcv = self._get_ohlcv_array()
        prices = self._get_current_prices()
        progress = self._state.step / max(self.config.max_episode_steps, 1)
        return self.obs_builder.build(
            ohlcv, self._state, prices, self._bh_baseline, progress
        )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one trading step.

        Args:
            action: np.ndarray of shape (n_instruments,), values in [-1, +1]

        Returns:
            obs, reward, terminated, truncated, info
        """
        assert self._state is not None, "Call reset() before step()"

        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        # Current prices (close of current bar, used for execution on next open)
        current_prices = self._get_current_prices()
        current_bar = self._aligned_data[self._current_idx]

        # Execute trades
        transaction_costs, position_changes = self._execute_trades(action, current_prices)

        # Advance to next bar
        self._current_idx += 1
        if self._current_idx < self.T:
            self._ohlcv_window.append(self._aligned_data[self._current_idx])
        self._state.step += 1

        # Update portfolio value
        new_prices = self._get_current_prices() if self._current_idx < self.T else current_prices
        prev_value = self._state.portfolio_value
        self._update_portfolio_value(new_prices)
        new_value = self._state.portfolio_value

        # Compute step return
        step_return = float((new_value - prev_value) / (prev_value + 1e-8))
        self._state.return_history.append(step_return)
        self._state.pnl_history.append(new_value - self.config.initial_capital)
        self._state.total_pnl = float(new_value - self.config.initial_capital)

        # Update peak and drawdown
        if new_value > self._state.peak_value:
            self._state.peak_value = new_value
        drawdown = float(1.0 - new_value / (self._state.peak_value + 1e-8))
        self._state.max_drawdown = max(self._state.max_drawdown, drawdown)

        # BH update
        self._bh_baseline.step(new_prices)

        # Compute reward
        reward, reward_info = self._reward_computer.compute(
            step_return, self._state, transaction_costs, position_changes, drawdown
        )

        # Check termination conditions
        terminated = False
        truncated = False

        if self._state.max_drawdown > self.config.max_drawdown_limit:
            terminated = True
            reward -= 10.0  # large penalty for blowup

        if self._state.step >= self.config.max_episode_steps:
            truncated = True

        if self._current_idx >= self.T - 1:
            truncated = True

        obs = self._build_observation() if not (terminated or truncated) else np.zeros(self.obs_dim, dtype=np.float32)

        # Build info dict
        info = {
            "step": self._state.step,
            "portfolio_value": float(new_value),
            "cash": float(self._state.cash),
            "positions": self._state.positions.tolist(),
            "drawdown": float(drawdown),
            "max_drawdown": float(self._state.max_drawdown),
            "step_return": float(step_return),
            "total_pnl": float(self._state.total_pnl),
            "transaction_costs": float(transaction_costs),
            "total_trades": int(self._state.total_trades),
            "win_rate": float(self._state.winning_trades / max(self._state.total_trades, 1)),
            **reward_info,
        }
        self._info_history.append(info)

        return obs, float(reward), terminated, truncated, info

    def _execute_trades(
        self, target_positions: np.ndarray, prices: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Map target position fractions to actual holdings.
        Returns (total_transaction_cost, position_changes).
        """
        total_cost = 0.0
        old_positions = self._state.positions.copy()
        position_changes = np.zeros(self.n_instruments, dtype=np.float32)

        for i, (instr, target) in enumerate(zip(self.instruments, target_positions)):
            current_pos = float(self._state.positions[i])
            delta = float(target) - current_pos

            if abs(delta) < 1e-4:
                continue

            price = float(prices[i])
            slippage = instr.slippage * abs(delta)
            tc = instr.transaction_cost * abs(delta)
            exec_price = price * (1.0 + np.sign(delta) * (slippage + tc))

            # Update cash
            notional = abs(delta) * self.config.initial_capital
            cost = notional * (slippage + tc)
            self._state.cash -= cost
            total_cost += cost

            # Track trade
            if abs(delta) > 0.01:
                self._state.total_trades += 1
                # Simple win/loss tracking
                prev_pos = current_pos
                if abs(prev_pos) > 1e-4:
                    prev_entry = float(self._state.avg_entry_prices[i])
                    pnl = prev_pos * (price - prev_entry) / (prev_entry + 1e-8)
                    if pnl > 0:
                        self._state.winning_trades += 1

            # Update avg entry price (FIFO approximation)
            if abs(float(target)) > 1e-4:
                old_pos = float(self._state.positions[i])
                if np.sign(float(target)) == np.sign(old_pos) and abs(old_pos) > 1e-4:
                    w_old = abs(old_pos) / (abs(old_pos) + abs(delta))
                    w_new = abs(delta) / (abs(old_pos) + abs(delta))
                    self._state.avg_entry_prices[i] = (
                        w_old * float(self._state.avg_entry_prices[i]) + w_new * exec_price
                    )
                else:
                    self._state.avg_entry_prices[i] = exec_price
            else:
                self._state.avg_entry_prices[i] = 0.0

            self._state.positions[i] = float(target)
            position_changes[i] = delta

        return total_cost, position_changes

    def _update_portfolio_value(self, current_prices: np.ndarray) -> None:
        """Recompute portfolio value from positions + cash."""
        position_value = 0.0
        for i in range(self.n_instruments):
            pos = float(self._state.positions[i])
            if abs(pos) > 1e-8:
                entry = float(self._state.avg_entry_prices[i])
                price = float(current_prices[i])
                notional = abs(pos) * self.config.initial_capital
                pnl = pos * (price - entry) / (entry + 1e-8) * self.config.initial_capital
                position_value += pnl

        self._state.portfolio_value = float(self._state.cash + position_value + self.config.initial_capital)

    def render(self, mode: str = "human") -> Optional[str]:
        """Render current state."""
        if self._state is None:
            return None
        s = (
            f"Step {self._state.step:4d} | "
            f"Value: ${self._state.portfolio_value:,.0f} | "
            f"PnL: ${self._state.total_pnl:+,.0f} | "
            f"DD: {self._state.max_drawdown:.1%} | "
            f"Pos: {self._state.positions}"
        )
        if mode == "human":
            print(s)
        return s

    def close(self) -> None:
        pass

    def get_episode_stats(self) -> Dict[str, Any]:
        """Return summary statistics for the completed episode."""
        if not self._state or not self._state.return_history:
            return {}
        returns = np.array(self._state.return_history)
        sharpe = float(returns.mean() / (returns.std() + 1e-8) * np.sqrt(252))
        ann_return = float(returns.mean() * 252)
        calmar = float(ann_return / (self._state.max_drawdown + 1e-4))
        win_rate = float(self._state.winning_trades / max(self._state.total_trades, 1))

        return {
            "sharpe": sharpe,
            "annualized_return": ann_return,
            "max_drawdown": float(self._state.max_drawdown),
            "calmar": calmar,
            "total_pnl": float(self._state.total_pnl),
            "total_trades": int(self._state.total_trades),
            "win_rate": win_rate,
            "final_value": float(self._state.portfolio_value),
        }


# ---------------------------------------------------------------------------
# Vectorized environment wrapper
# ---------------------------------------------------------------------------

class VecTradingEnv:
    """
    Vectorized wrapper: runs N independent TradingEnv instances in parallel
    (using Python threads or multiprocessing).
    """

    def __init__(self, env_fns: List[callable], use_multiprocessing: bool = False):
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)
        self.obs_dim = self.envs[0].obs_dim
        self.act_dim = self.envs[0].act_dim
        self._dones = np.zeros(self.n_envs, dtype=bool)

    def reset(self) -> np.ndarray:
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        self._dones[:] = False
        return np.stack(obs_list, axis=0)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Args:
            actions: (n_envs, act_dim)
        Returns:
            obs: (n_envs, obs_dim)
            rewards: (n_envs,)
            dones: (n_envs,)
            infos: list of info dicts
        """
        obs_list, reward_list, done_list, info_list = [], [], [], []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if self._dones[i]:
                obs, _ = env.reset()
                reward = 0.0
                done = False
                info = {}
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                self._dones[i] = done
                if done:
                    # Auto-reset
                    obs, _ = env.reset()
                    self._dones[i] = False

            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return (
            np.stack(obs_list, axis=0),
            np.array(reward_list, dtype=np.float32),
            np.array(done_list, dtype=bool),
            info_list,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()


# ---------------------------------------------------------------------------
# Regime-aware environment (wraps TradingEnv, samples different market regimes)
# ---------------------------------------------------------------------------

class RegimeTradingEnv(TradingEnv):
    """
    Extends TradingEnv with regime labeling and regime-conditioned resets.
    Regimes: 0=bull, 1=bear, 2=sideways, 3=high_vol, 4=low_vol
    """

    REGIME_NAMES = ["bull", "bear", "sideways", "high_vol", "low_vol"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._regimes = self._label_regimes()
        self._current_regime: int = 0

    def _label_regimes(self) -> np.ndarray:
        """Label each bar with a regime index."""
        regimes = np.zeros(self.T, dtype=np.int32)

        # Use first instrument's close prices for regime detection
        closes = self._aligned_data[:, 0, 3]
        volumes = self._aligned_data[:, 0, 4]

        window = 20
        for t in range(window, self.T):
            start = t - window
            price_change = (closes[t] - closes[start]) / (closes[start] + 1e-8)
            vol_std = closes[start:t].std() / (closes[start:t].mean() + 1e-8)

            if price_change > 0.05:
                regime = 0  # bull
            elif price_change < -0.05:
                regime = 1  # bear
            elif vol_std > 0.03:
                regime = 3  # high_vol
            elif vol_std < 0.005:
                regime = 4  # low_vol
            else:
                regime = 2  # sideways
            regimes[t] = regime

        return regimes

    def reset(self, regime: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = super().reset(**kwargs)
        self._current_regime = int(self._regimes[self._current_idx])
        if regime is not None:
            # Try to find an episode starting in the requested regime
            candidates = np.where(self._regimes == regime)[0]
            if len(candidates) > 0:
                valid = candidates[candidates < self.T - self.config.max_episode_steps]
                if len(valid) > 0:
                    start = int(self.np_random.choice(valid))
                    self._episode_start_idx = start
                    self._current_idx = start
                    self._current_regime = regime
                    obs = self._build_observation()
        info["regime"] = self._current_regime
        info["regime_name"] = self.REGIME_NAMES[self._current_regime]
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        if self._current_idx < self.T:
            self._current_regime = int(self._regimes[self._current_idx])
        info["regime"] = self._current_regime
        info["regime_name"] = self.REGIME_NAMES[self._current_regime]
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_trading_env(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    instruments: Optional[List[Instrument]] = None,
    config: Optional[TradingConfig] = None,
    regime_aware: bool = False,
) -> TradingEnv:
    """Factory function for creating trading environments."""
    cls = RegimeTradingEnv if regime_aware else TradingEnv
    return cls(data=data, instruments=instruments, config=config)


def make_vec_env(
    data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    n_envs: int = 8,
    instruments: Optional[List[Instrument]] = None,
    config: Optional[TradingConfig] = None,
    regime_aware: bool = False,
    seeds: Optional[List[int]] = None,
) -> VecTradingEnv:
    """Create a vectorized set of trading environments."""
    def _make_fn(seed: int):
        def fn():
            env = make_trading_env(data, instruments, config, regime_aware)
            env.seed(seed)
            return env
        return fn

    seeds = seeds or list(range(n_envs))
    env_fns = [_make_fn(s) for s in seeds]
    return VecTradingEnv(env_fns)


def generate_synthetic_data(
    n_assets: int = 3,
    n_days: int = 1000,
    seed: int = 42,
    regime_changes: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic OHLCV data for testing.
    Optionally adds regime changes (trending, mean-reverting, high-vol).
    """
    rng = np.random.default_rng(seed)
    symbols = [f"ASSET_{i}" for i in range(n_assets)]
    result = {}

    for sym in symbols:
        prices = np.zeros(n_days)
        prices[0] = rng.uniform(50, 200)

        for t in range(1, n_days):
            if regime_changes:
                # Rotate regimes every ~100 bars
                regime = (t // 100) % 5
                if regime == 0:    # bull
                    drift, vol = 0.001, 0.01
                elif regime == 1:  # bear
                    drift, vol = -0.001, 0.012
                elif regime == 2:  # sideways
                    drift, vol = 0.0, 0.008
                elif regime == 3:  # high_vol
                    drift, vol = 0.0, 0.025
                else:              # low_vol
                    drift, vol = 0.0002, 0.004
            else:
                drift, vol = 0.0003, 0.01

            prices[t] = prices[t - 1] * np.exp(drift + vol * rng.standard_normal())

        # Generate OHLCV
        opens  = prices * (1 + rng.uniform(-0.002, 0.002, n_days))
        highs  = prices * (1 + rng.uniform(0.001, 0.015, n_days))
        lows   = prices * (1 - rng.uniform(0.001, 0.015, n_days))
        closes = prices
        volumes = rng.lognormal(10, 1, n_days).astype(np.float32)

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes
        })
        result[sym] = df

    return result


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating synthetic data...")
    data = generate_synthetic_data(n_assets=2, n_days=500)

    config = TradingConfig(
        initial_capital=100_000,
        max_episode_steps=50,
        window_size=30,
        reward_type="sharpe",
        use_bh_features=True,
    )

    env = make_trading_env(data, config=config, regime_aware=True)
    obs, info = env.reset()
    print(f"Obs dim: {len(obs)}, expected: {env.obs_dim}")

    total_reward = 0.0
    for step in range(50):
        action = np.random.uniform(-1, 1, env.act_dim)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    stats = env.get_episode_stats()
    print(f"Episode stats: {stats}")
    print(f"Total reward: {total_reward:.4f}")
    print("TradingEnv self-test passed.")
