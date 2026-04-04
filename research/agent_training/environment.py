"""
research/agent_training/environment.py

Gym-compatible trading environments for agent training.

Observation space (14 features):
    [price_pct_change, tf_score, mass, atr_norm, regime_encoded,
     equity_ratio, position_ratio, unrealized_pnl_pct,
     rolling_vol_5, rolling_vol_20, momentum_5, momentum_20,
     ensemble_signal, bh_active]

Action space: continuous float in [-1, 1]
    negative => short/reduce position
    positive => long/increase position
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBS_DIM = 14
OBS_LABELS: list[str] = [
    "price_pct_change",
    "tf_score",
    "mass",
    "atr_norm",
    "regime_encoded",
    "equity_ratio",
    "position_ratio",
    "unrealized_pnl_pct",
    "rolling_vol_5",
    "rolling_vol_20",
    "momentum_5",
    "momentum_20",
    "ensemble_signal",
    "bh_active",
]

REGIME_MAP: dict[str, float] = {
    "BULL": 1.0,
    "BEAR": -1.0,
    "SIDEWAYS": 0.0,
    "HIGH_VOLATILITY": 0.5,
    "UNKNOWN": 0.0,
}


# ---------------------------------------------------------------------------
# EnvironmentConfig
# ---------------------------------------------------------------------------


@dataclass
class EnvironmentConfig:
    """All hyperparameters for TradingEnvironment."""

    starting_equity: float = 1_000_000.0
    max_position_frac: float = 0.75
    transaction_cost: float = 0.0002
    reward_shaping: str = "sharpe"          # sharpe | sortino | log_return | calmar
    reward_window: int = 30
    max_steps: int = 2000
    min_steps: int = 50
    clip_obs: bool = True
    obs_clip_range: float = 5.0
    penalty_drawdown: float = 0.1          # extra penalty per unit max drawdown
    penalty_holding: float = 0.0001        # tiny penalty per step for large positions
    slippage_model: str = "fixed"          # fixed | proportional | square_root
    slippage_bps: float = 1.0              # basis points
    margin_requirement: float = 0.5        # fraction of position value required as margin
    liquidation_threshold: float = 0.05    # liquidate if equity < 5% of start
    curriculum_difficulty: float = 1.0     # 0.0 = easy (trending), 1.0 = full difficulty
    use_sortino_mar: float = 0.0
    vol_target: float = 0.15               # annualised vol target for position sizing
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# EpisodeStats
# ---------------------------------------------------------------------------


@dataclass
class EpisodeStats:
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    n_trades: int
    win_rate: float
    avg_profit_per_trade: float
    total_transaction_cost: float
    equity_curve: np.ndarray
    returns: np.ndarray
    actions: np.ndarray
    info_list: list[dict]


def episode_stats(transitions: list[dict]) -> EpisodeStats:
    """
    Compute EpisodeStats from a list of transition dicts produced by
    TradingEnvironment.step(). Each dict must contain keys:
        equity, reward, action, info (dict with 'trade_closed', 'pnl')
    """
    equities = np.array([t["equity"] for t in transitions], dtype=np.float64)
    rewards = np.array([t["reward"] for t in transitions], dtype=np.float64)
    actions = np.array([t["action"] for t in transitions], dtype=np.float64)

    returns = np.diff(equities) / equities[:-1]
    if len(returns) == 0:
        returns = np.zeros(1)

    total_return = float((equities[-1] - equities[0]) / equities[0])
    ann_factor = math.sqrt(252)
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns)) + 1e-12
    sharpe = float(mean_r / std_r * ann_factor)

    downside = returns[returns < 0.0]
    std_d = float(np.std(downside)) + 1e-12 if len(downside) > 0 else std_r
    sortino = float(mean_r / std_d * ann_factor)

    running_max = np.maximum.accumulate(equities)
    drawdowns = (running_max - equities) / running_max
    max_dd = float(np.max(drawdowns)) + 1e-12
    calmar = float(total_return / max_dd) if max_dd > 0 else 0.0

    trade_pnls = [
        t["info"].get("trade_pnl", 0.0)
        for t in transitions
        if t["info"].get("trade_closed", False)
    ]
    n_trades = len(trade_pnls)
    win_rate = float(sum(1 for p in trade_pnls if p > 0) / n_trades) if n_trades else 0.0
    avg_profit = float(np.mean(trade_pnls)) if trade_pnls else 0.0
    total_cost = float(sum(t["info"].get("transaction_cost", 0.0) for t in transitions))

    return EpisodeStats(
        total_return=total_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        n_trades=n_trades,
        win_rate=win_rate,
        avg_profit_per_trade=avg_profit,
        total_transaction_cost=total_cost,
        equity_curve=equities,
        returns=returns,
        actions=actions,
        info_list=[t["info"] for t in transitions],
    )


# ---------------------------------------------------------------------------
# RewardShaper
# ---------------------------------------------------------------------------


class RewardShaper:
    """
    Collection of reward shaping functions for reinforcement learning.

    All functions return a scalar float. They are designed to be called
    once per environment step with a rolling window of recent returns.
    """

    @staticmethod
    def sharpe_reward(returns_window: np.ndarray) -> float:
        """
        Risk-adjusted Sharpe-based reward over the supplied window.

        Returns the ratio mean/std scaled to [-1, 1] via tanh.
        """
        if len(returns_window) < 2:
            return 0.0
        mu = float(np.mean(returns_window))
        sigma = float(np.std(returns_window)) + 1e-10
        return float(np.tanh(mu / sigma * math.sqrt(len(returns_window))))

    @staticmethod
    def sortino_reward(returns_window: np.ndarray, mar: float = 0.0) -> float:
        """
        Downside-deviation-adjusted reward (Sortino ratio proxy).

        Args:
            returns_window: 1-D array of recent per-step returns.
            mar: Minimum acceptable return threshold (per step).

        Returns:
            Scalar in approximately [-1, 1].
        """
        if len(returns_window) < 2:
            return 0.0
        excess = returns_window - mar
        downside = excess[excess < 0.0]
        if len(downside) == 0:
            return float(np.tanh(np.mean(excess) * 10.0))
        std_d = float(np.std(downside)) + 1e-10
        mu = float(np.mean(excess))
        return float(np.tanh(mu / std_d * math.sqrt(len(returns_window))))

    @staticmethod
    def log_return_reward(pnl: float, equity: float) -> float:
        """
        Log-utility reward: log(1 + pnl/equity).

        Provides diminishing marginal utility for large gains and
        strong penalties for losses approaching ruin.
        """
        if equity <= 0.0:
            return -1.0
        ratio = pnl / equity
        if 1.0 + ratio <= 0.0:
            return -1.0
        return float(math.log1p(ratio))

    @staticmethod
    def calmar_reward(equity_curve: np.ndarray) -> float:
        """
        Calmar-ratio-based reward over the supplied equity curve.

        Returns reward in [-1, 1] via tanh.
        """
        if len(equity_curve) < 2:
            return 0.0
        total_ret = float((equity_curve[-1] - equity_curve[0]) / (equity_curve[0] + 1e-12))
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / (running_max + 1e-12)
        max_dd = float(np.max(drawdowns)) + 1e-12
        calmar = total_ret / max_dd
        return float(np.tanh(calmar * 0.5))

    @staticmethod
    def composite_reward(
        returns_window: np.ndarray,
        pnl: float,
        equity: float,
        equity_curve: np.ndarray,
        weights: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1),
    ) -> float:
        """
        Weighted combination of all four reward signals.

        Args:
            returns_window: Recent returns for Sharpe/Sortino.
            pnl: Step P&L for log-return reward.
            equity: Current equity for log-return reward.
            equity_curve: Full episode equity curve for Calmar.
            weights: (sharpe_w, sortino_w, log_w, calmar_w).

        Returns:
            Scalar composite reward.
        """
        ws, wso, wl, wc = weights
        r_sharpe = RewardShaper.sharpe_reward(returns_window)
        r_sortino = RewardShaper.sortino_reward(returns_window)
        r_log = RewardShaper.log_return_reward(pnl, equity)
        r_calmar = RewardShaper.calmar_reward(equity_curve)
        return float(ws * r_sharpe + wso * r_sortino + wl * r_log + wc * r_calmar)


# ---------------------------------------------------------------------------
# Slippage models
# ---------------------------------------------------------------------------


def _compute_slippage(
    price: float,
    trade_size: float,
    model: str,
    bps: float,
) -> float:
    """Return slippage cost as a fraction of position value."""
    if model == "fixed":
        return abs(trade_size) * price * bps / 10_000.0
    elif model == "proportional":
        return abs(trade_size) * price * bps / 10_000.0 * (1.0 + abs(trade_size) * 0.01)
    elif model == "square_root":
        # Market-impact square-root model
        return abs(trade_size) * price * bps / 10_000.0 * math.sqrt(abs(trade_size) + 1e-6)
    return 0.0


# ---------------------------------------------------------------------------
# TradingEnvironment
# ---------------------------------------------------------------------------


class TradingEnvironment:
    """
    Gym-compatible single-instrument trading environment.

    price_data: pd.DataFrame with columns including 'close' (required)
        and optionally 'open', 'high', 'low', 'volume', 'atr', 'tf_score',
        'mass', 'regime', 'delta_score', 'ensemble_signal', 'bh_active'.
    features: np.ndarray of shape (T, F) — pre-computed feature matrix
        aligned row-by-row with price_data.
    """

    metadata: dict = {"render.modes": ["human", "ansi"]}

    def __init__(
        self,
        price_data: pd.DataFrame,
        features: np.ndarray,
        starting_equity: float = 1_000_000.0,
        max_position_frac: float = 0.75,
        transaction_cost: float = 0.0002,
        config: Optional[EnvironmentConfig] = None,
    ) -> None:
        if config is None:
            config = EnvironmentConfig(
                starting_equity=starting_equity,
                max_position_frac=max_position_frac,
                transaction_cost=transaction_cost,
            )
        self.cfg = config
        self.price_data = price_data.reset_index(drop=True)
        self.features = np.asarray(features, dtype=np.float64)

        # Validate shapes
        n_price = len(self.price_data)
        n_feat = len(self.features)
        if n_price != n_feat:
            raise ValueError(
                f"price_data has {n_price} rows but features has {n_feat} rows."
            )
        if n_price < 30:
            raise ValueError("Need at least 30 rows of price data.")

        self.n_steps = n_price
        self.obs_dim = OBS_DIM
        self.action_dim = 1

        # Precompute log returns for the whole dataset
        closes = self.price_data["close"].values.astype(np.float64)
        self._closes = closes
        self._log_returns = np.zeros(n_price)
        self._log_returns[1:] = np.log(closes[1:] / (closes[:-1] + 1e-12))

        # Precompute ATR normalised
        if "atr" in self.price_data.columns:
            self._atr_norm = (
                self.price_data["atr"].values / (closes + 1e-12)
            ).astype(np.float64)
        else:
            # Rough proxy: rolling std of log returns * sqrt(252)
            self._atr_norm = _rolling_std(self._log_returns, 14)

        # Rolling volatilities (precomputed)
        self._vol5 = _rolling_std(self._log_returns, 5)
        self._vol20 = _rolling_std(self._log_returns, 20)

        # Momentum (log return over window)
        self._mom5 = _rolling_sum(self._log_returns, 5)
        self._mom20 = _rolling_sum(self._log_returns, 20)

        # Regime encoding
        if "regime" in self.price_data.columns:
            self._regime = np.array(
                [REGIME_MAP.get(str(r).upper(), 0.0) for r in self.price_data["regime"]],
                dtype=np.float64,
            )
        else:
            self._regime = np.zeros(n_price)

        # Optional BH signals
        self._tf_score = self._get_col("tf_score", 0.0)
        self._mass = self._get_col("mass", 0.0)
        self._ensemble_signal = self._get_col("ensemble_signal", 0.0)
        self._bh_active = self._get_col("bh_active", 0.0)
        self._delta_score = self._get_col("delta_score", 0.0)

        # RNG
        seed = self.cfg.seed
        self._rng = np.random.default_rng(seed)

        # Episode state (initialised in reset)
        self._step_idx: int = 0
        self._start_idx: int = 0
        self._equity: float = 0.0
        self._position: float = 0.0   # signed position in units of the asset
        self._entry_price: float = 0.0
        self._returns_window: deque = deque(maxlen=self.cfg.reward_window)
        self._equity_curve: list[float] = []
        self._n_trades: int = 0
        self._total_cost: float = 0.0
        self._max_equity: float = 0.0
        self._prev_equity: float = 0.0
        self._reward_shaper = RewardShaper()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_col(self, col: str, default: float) -> np.ndarray:
        if col in self.price_data.columns:
            return self.price_data[col].fillna(default).values.astype(np.float64)
        return np.full(len(self.price_data), default, dtype=np.float64)

    def _build_obs(self) -> np.ndarray:
        i = self._step_idx
        price = self._closes[i]

        price_pct = float(self._log_returns[i])
        tf_score = float(self._tf_score[i])
        mass = float(self._mass[i])
        atr_norm = float(self._atr_norm[i])
        regime_enc = float(self._regime[i])

        equity_ratio = float(self._equity / self.cfg.starting_equity)
        position_value = self._position * price
        position_ratio = float(position_value / (self._equity + 1e-12))
        position_ratio = float(np.clip(position_ratio, -1.0, 1.0))

        if self._position != 0.0 and self._entry_price > 0.0:
            unrealized_pnl = self._position * (price - self._entry_price)
            unrealized_pnl_pct = float(unrealized_pnl / (self._equity + 1e-12))
        else:
            unrealized_pnl_pct = 0.0

        vol5 = float(self._vol5[i])
        vol20 = float(self._vol20[i])
        mom5 = float(self._mom5[i])
        mom20 = float(self._mom20[i])
        ensemble = float(self._ensemble_signal[i])
        bh_active = float(self._bh_active[i])

        obs = np.array(
            [
                price_pct,
                tf_score,
                mass,
                atr_norm,
                regime_enc,
                equity_ratio,
                position_ratio,
                unrealized_pnl_pct,
                vol5,
                vol20,
                mom5,
                mom20,
                ensemble,
                bh_active,
            ],
            dtype=np.float64,
        )

        if self.cfg.clip_obs:
            obs = np.clip(obs, -self.cfg.obs_clip_range, self.cfg.obs_clip_range)

        return obs

    def _compute_reward(self, step_pnl: float) -> float:
        returns_arr = np.array(self._returns_window, dtype=np.float64)
        method = self.cfg.reward_shaping

        if method == "sharpe":
            reward = self._reward_shaper.sharpe_reward(returns_arr)
        elif method == "sortino":
            reward = self._reward_shaper.sortino_reward(
                returns_arr, mar=self.cfg.use_sortino_mar
            )
        elif method == "log_return":
            reward = self._reward_shaper.log_return_reward(step_pnl, self._equity)
        elif method == "calmar":
            reward = self._reward_shaper.calmar_reward(
                np.array(self._equity_curve, dtype=np.float64)
            )
        else:
            reward = self._reward_shaper.sharpe_reward(returns_arr)

        # Drawdown penalty
        if self._max_equity > 0:
            dd = (self._max_equity - self._equity) / self._max_equity
            reward -= dd * self.cfg.penalty_drawdown

        # Holding penalty for large positions
        abs_pos_ratio = abs(self._position * self._closes[self._step_idx]) / (
            self._equity + 1e-12
        )
        reward -= abs_pos_ratio * self.cfg.penalty_holding

        return float(reward)

    def _execute_trade(
        self, action: float, price: float
    ) -> tuple[float, float, dict]:
        """
        Execute a trade given a continuous action in [-1, 1].

        Returns: (pnl, cost, trade_info)
        """
        max_pos_value = self.cfg.max_position_frac * self._equity
        target_pos = action * max_pos_value / (price + 1e-12)

        delta_pos = target_pos - self._position

        trade_closed = False
        trade_pnl = 0.0

        # Transaction cost + slippage
        cost = abs(delta_pos) * price * self.cfg.transaction_cost
        cost += _compute_slippage(price, delta_pos, self.cfg.slippage_model, self.cfg.slippage_bps)

        if abs(delta_pos) > 1e-10:
            # Closing or reducing position
            if (self._position > 0 and delta_pos < 0) or (
                self._position < 0 and delta_pos > 0
            ):
                closed_units = min(abs(delta_pos), abs(self._position)) * np.sign(delta_pos) * -1
                trade_pnl = closed_units * (price - self._entry_price)
                if abs(delta_pos) >= abs(self._position):
                    trade_closed = True
            # Update position
            prev_pos = self._position
            self._position = target_pos

            if trade_closed or prev_pos == 0.0:
                # Starting new position or fully closed
                if abs(self._position) > 1e-10:
                    self._entry_price = price
                else:
                    self._entry_price = 0.0
            elif prev_pos == 0.0:
                self._entry_price = price
            elif np.sign(self._position) != np.sign(prev_pos):
                # Flipped sides
                self._entry_price = price
            else:
                # Averaging in
                if abs(self._position) > abs(prev_pos):
                    # Adding to position — compute average entry
                    total_cost_basis = prev_pos * self._entry_price + delta_pos * price
                    self._entry_price = total_cost_basis / (self._position + 1e-12)

            self._n_trades += 1

        self._total_cost += cost

        return trade_pnl, cost, {
            "trade_closed": trade_closed,
            "trade_pnl": trade_pnl,
            "transaction_cost": cost,
            "delta_position": delta_pos,
            "target_position": target_pos,
        }

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self,
        start_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Reset the environment for a new episode.

        Args:
            start_idx: Force a specific start index (useful for walk-forward).
                       If None, a random start is chosen, leaving room for
                       at least min_steps + reward_window steps.

        Returns:
            Initial observation as np.ndarray of shape (OBS_DIM,).
        """
        margin = self.cfg.reward_window + 20
        max_start = self.n_steps - self.cfg.min_steps - margin

        if start_idx is not None:
            self._start_idx = int(np.clip(start_idx, margin, max_start))
        else:
            self._start_idx = int(self._rng.integers(margin, max(margin + 1, max_start)))

        self._step_idx = self._start_idx
        self._equity = self.cfg.starting_equity
        self._prev_equity = self._equity
        self._max_equity = self._equity
        self._position = 0.0
        self._entry_price = 0.0
        self._returns_window = deque(maxlen=self.cfg.reward_window)
        self._equity_curve = [self._equity]
        self._n_trades = 0
        self._total_cost = 0.0

        # Pre-fill returns window
        start = max(0, self._start_idx - self.cfg.reward_window)
        for i in range(start, self._start_idx):
            self._returns_window.append(float(self._log_returns[i]))

        return self._build_obs()

    def step(
        self, action: float
    ) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute one environment step.

        Args:
            action: Float in [-1, 1]. Positive = long, negative = short.

        Returns:
            observation: np.ndarray of shape (OBS_DIM,)
            reward: float
            done: bool
            info: dict with diagnostic information
        """
        action = float(np.clip(action, -1.0, 1.0))
        price = float(self._closes[self._step_idx])

        # Execute trade
        trade_pnl, cost, trade_info = self._execute_trade(action, price)

        # Advance to next step
        self._step_idx += 1
        done = self._step_idx >= self.n_steps or self._step_idx >= (
            self._start_idx + self.cfg.max_steps
        )

        # Mark-to-market position at new price
        if not done:
            new_price = float(self._closes[self._step_idx])
        else:
            new_price = price

        unrealized = self._position * (new_price - price)
        step_pnl = trade_pnl + unrealized - cost

        self._prev_equity = self._equity
        self._equity = max(0.0, self._equity + step_pnl)
        self._max_equity = max(self._max_equity, self._equity)
        self._equity_curve.append(self._equity)

        # Step return for reward window
        step_return = (self._equity - self._prev_equity) / (self._prev_equity + 1e-12)
        self._returns_window.append(step_return)

        # Liquidation check
        if self._equity < self.cfg.liquidation_threshold * self.cfg.starting_equity:
            done = True

        reward = self._compute_reward(step_pnl)

        info: dict[str, Any] = {
            **trade_info,
            "equity": self._equity,
            "position": self._position,
            "price": new_price,
            "step_idx": self._step_idx,
            "step_return": step_return,
            "n_trades": self._n_trades,
            "total_cost": self._total_cost,
        }

        if done:
            obs = self._build_obs()  # final obs at current step
        else:
            obs = self._build_obs()

        return obs, reward, done, info

    def render(self, mode: str = "human") -> Optional[str]:
        i = self._step_idx
        price = float(self._closes[min(i, self.n_steps - 1)])
        msg = (
            f"Step {i} | Price {price:.4f} | "
            f"Equity {self._equity:.2f} | "
            f"Position {self._position:.4f}"
        )
        if mode == "ansi":
            return msg
        print(msg)
        return None

    def close(self) -> None:
        pass

    @property
    def observation_space_shape(self) -> tuple[int, ...]:
        return (OBS_DIM,)

    @property
    def action_space_bounds(self) -> tuple[float, float]:
        return (-1.0, 1.0)

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self.cfg.seed = seed

    def get_equity_curve(self) -> np.ndarray:
        return np.array(self._equity_curve, dtype=np.float64)

    def get_current_equity(self) -> float:
        return self._equity

    def get_current_position(self) -> float:
        return self._position


# ---------------------------------------------------------------------------
# MultiInstrumentEnvironment
# ---------------------------------------------------------------------------


@dataclass
class InstrumentSpec:
    name: str
    price_data: pd.DataFrame
    features: np.ndarray
    weight_limit: float = 1.0  # max fraction of portfolio in this instrument


class MultiInstrumentEnvironment:
    """
    Simultaneous trading across N instruments.

    Observation: concatenation of per-instrument observations + portfolio-level stats.
    Action: np.ndarray of shape (N,) in [-1, 1], one per instrument.

    Portfolio-level obs appended (4 extra dims):
        [portfolio_return, portfolio_vol, n_active_positions, total_drawdown]
    """

    def __init__(
        self,
        instruments: list[InstrumentSpec],
        starting_equity: float = 1_000_000.0,
        max_position_frac: float = 0.30,
        transaction_cost: float = 0.0002,
        config: Optional[EnvironmentConfig] = None,
    ) -> None:
        if len(instruments) < 2:
            raise ValueError("MultiInstrumentEnvironment requires at least 2 instruments.")

        self.instruments = instruments
        self.n_instruments = len(instruments)

        if config is None:
            config = EnvironmentConfig(
                starting_equity=starting_equity,
                max_position_frac=max_position_frac,
                transaction_cost=transaction_cost,
            )
        self.cfg = config

        # Per-instrument equity slices (start equally weighted)
        self._per_instrument_equity = starting_equity / self.n_instruments

        # Build sub-environments
        self._envs: list[TradingEnvironment] = []
        for spec in instruments:
            env_cfg = EnvironmentConfig(
                starting_equity=self._per_instrument_equity,
                max_position_frac=spec.weight_limit,
                transaction_cost=transaction_cost,
                reward_shaping=config.reward_shaping,
                max_steps=config.max_steps,
                clip_obs=config.clip_obs,
                seed=config.seed,
            )
            self._envs.append(
                TradingEnvironment(
                    price_data=spec.price_data,
                    features=spec.features,
                    config=env_cfg,
                )
            )

        self.obs_dim = OBS_DIM * self.n_instruments + 4
        self.action_dim = self.n_instruments

        self._portfolio_returns: deque = deque(maxlen=30)
        self._portfolio_equity_curve: list[float] = []
        self._max_portfolio_equity: float = starting_equity

    def reset(self) -> np.ndarray:
        per_obs = []
        for env in self._envs:
            obs = env.reset()
            per_obs.append(obs)

        self._portfolio_returns = deque(maxlen=30)
        equity_now = sum(e.get_current_equity() for e in self._envs)
        self._portfolio_equity_curve = [equity_now]
        self._max_portfolio_equity = equity_now

        portfolio_obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return np.concatenate(per_obs + [portfolio_obs])

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict]:
        """
        Args:
            actions: np.ndarray of shape (N,) in [-1, 1].

        Returns:
            (observation, reward, done, info)
        """
        actions = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        if actions.shape[0] != self.n_instruments:
            raise ValueError(
                f"Expected {self.n_instruments} actions, got {actions.shape[0]}."
            )

        per_obs = []
        per_rewards = []
        per_infos = []
        any_done = False

        for i, (env, act) in enumerate(zip(self._envs, actions)):
            obs, reward, done, info = env.step(float(act))
            per_obs.append(obs)
            per_rewards.append(reward)
            per_infos.append(info)
            if done:
                any_done = True

        # Portfolio-level stats
        equity_now = sum(e.get_current_equity() for e in self._envs)
        self._max_portfolio_equity = max(self._max_portfolio_equity, equity_now)

        prev_equity = self._portfolio_equity_curve[-1] if self._portfolio_equity_curve else equity_now
        port_return = (equity_now - prev_equity) / (prev_equity + 1e-12)
        self._portfolio_returns.append(port_return)
        self._portfolio_equity_curve.append(equity_now)

        port_vol = float(np.std(list(self._portfolio_returns))) if len(self._portfolio_returns) > 1 else 0.0
        n_active = sum(1 for e in self._envs if abs(e.get_current_position()) > 1e-8)
        total_dd = float(
            (self._max_portfolio_equity - equity_now) / (self._max_portfolio_equity + 1e-12)
        )

        portfolio_obs = np.array(
            [port_return, port_vol, float(n_active), total_dd], dtype=np.float64
        )
        full_obs = np.concatenate(per_obs + [portfolio_obs])

        # Portfolio reward = mean of individual + diversification bonus
        mean_reward = float(np.mean(per_rewards))
        if n_active > 1:
            diversification_bonus = 0.01 * math.log(n_active)
        else:
            diversification_bonus = 0.0
        portfolio_reward = mean_reward + diversification_bonus

        info = {
            "per_instrument": per_infos,
            "portfolio_equity": equity_now,
            "portfolio_return": port_return,
            "n_active": n_active,
            "total_drawdown": total_dd,
        }

        return full_obs, portfolio_reward, any_done, info

    def render(self, mode: str = "human") -> None:
        for i, env in enumerate(self._envs):
            name = self.instruments[i].name
            env_str = env.render(mode="ansi") or ""
            if mode == "human":
                print(f"[{name}] {env_str}")

    def close(self) -> None:
        for env in self._envs:
            env.close()

    @property
    def observation_space_shape(self) -> tuple[int, ...]:
        return (self.obs_dim,)

    @property
    def action_space_shape(self) -> tuple[int, ...]:
        return (self.action_dim,)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling standard deviation, filling initial values with 0."""
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)
    for i in range(window, n):
        result[i] = float(np.std(arr[i - window : i]))
    # Fill head with expanding window
    for i in range(1, min(window, n)):
        result[i] = float(np.std(arr[: i + 1]))
    return result


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling sum, filling initial values with expanding sum."""
    n = len(arr)
    result = np.zeros(n, dtype=np.float64)
    for i in range(window, n):
        result[i] = float(np.sum(arr[i - window : i]))
    for i in range(1, min(window, n)):
        result[i] = float(np.sum(arr[: i + 1]))
    return result


def make_environment(
    price_df: pd.DataFrame,
    features: np.ndarray,
    config: Optional[EnvironmentConfig] = None,
) -> TradingEnvironment:
    """
    Factory helper. Returns a TradingEnvironment with the given data and config.
    """
    return TradingEnvironment(
        price_data=price_df,
        features=features,
        config=config or EnvironmentConfig(),
    )


def random_episode(
    env: TradingEnvironment,
    n_steps: int = 200,
    seed: Optional[int] = None,
) -> list[dict]:
    """
    Run a random-action episode for baseline comparison.

    Returns list of transition dicts compatible with episode_stats().
    """
    rng = np.random.default_rng(seed)
    obs = env.reset()
    transitions = []
    for _ in range(n_steps):
        action = float(rng.uniform(-1.0, 1.0))
        next_obs, reward, done, info = env.step(action)
        transitions.append(
            {
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "equity": info["equity"],
                "info": info,
            }
        )
        obs = next_obs
        if done:
            break
    return transitions


def compute_benchmark_returns(
    price_data: pd.DataFrame,
    start_idx: int,
    end_idx: int,
) -> np.ndarray:
    """
    Compute buy-and-hold log returns for the given range.
    """
    closes = price_data["close"].values.astype(np.float64)
    segment = closes[start_idx : end_idx + 1]
    if len(segment) < 2:
        return np.zeros(1)
    return np.log(segment[1:] / (segment[:-1] + 1e-12))


def difficulty_schedule(
    episode: int,
    max_episodes: int,
    min_difficulty: float = 0.1,
    max_difficulty: float = 1.0,
) -> float:
    """
    Curriculum learning difficulty schedule.

    Returns a difficulty scalar in [min_difficulty, max_difficulty].
    Ramps up smoothly from easy to hard over the course of training.
    """
    t = float(episode) / max(1, max_episodes)
    difficulty = min_difficulty + (max_difficulty - min_difficulty) * (t ** 0.5)
    return float(np.clip(difficulty, min_difficulty, max_difficulty))
