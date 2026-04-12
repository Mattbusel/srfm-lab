"""
chronos_env_bridge.py
=====================
Gymnasium / PettingZoo-compatible wrapper around the Chronos limit-order-book
(LOB) simulator.  The bridge reads Chronos CSV output, steps through the data
tick-by-tick, exposes rich order-book observation tensors, a structured action
space (limit / market orders with size), and a fully configurable reward
pipeline.  It also supports vectorised multi-environment execution (VecEnv)
for data-parallel MARL training.

Dependencies: gymnasium, pettingzoo, numpy, pandas, torch
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import enum
import io
import itertools
import logging
import math
import os
import pathlib
import random
import time
import warnings
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym  # type: ignore
    from gym import spaces  # type: ignore

try:
    from pettingzoo import AECEnv, ParallelEnv
    from pettingzoo.utils import agent_selector, wrappers
except ImportError:  # pragma: no cover
    AECEnv = object  # type: ignore
    ParallelEnv = object  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------

DEFAULT_LOB_DEPTH: int = 10          # price levels shown in observation
DEFAULT_MAX_INVENTORY: float = 1000.0
DEFAULT_TICK_SIZE: float = 0.01
DEFAULT_LOT_SIZE: float = 1.0
DEFAULT_MAX_ORDER_SIZE: int = 50
DEFAULT_EPISODE_LEN: int = 2000      # ticks
PRICE_NORM_SCALE: float = 1_000.0
VOL_NORM_SCALE: float = 10_000.0

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class OrderType(enum.IntEnum):
    LIMIT = 0
    MARKET = 1
    CANCEL = 2
    NOP = 3


class OrderSide(enum.IntEnum):
    BUY = 0
    SELL = 1


class RewardMode(enum.IntEnum):
    REALIZED_PNL = 0
    MARK_TO_MARKET = 1
    SHAPED_SHARPE = 2
    ADVERSARIAL = 3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class LOBSnapshot:
    """A single tick snapshot of the limit order book."""
    timestamp: float
    mid_price: float
    best_bid: float
    best_ask: float
    spread: float
    bid_prices: np.ndarray   # shape (depth,)
    bid_volumes: np.ndarray  # shape (depth,)
    ask_prices: np.ndarray   # shape (depth,)
    ask_volumes: np.ndarray  # shape (depth,)
    last_trade_price: float = 0.0
    last_trade_volume: float = 0.0
    cumulative_volume: float = 0.0
    vwap: float = 0.0
    volatility_est: float = 0.0

    @property
    def imbalance(self) -> float:
        """Order-book imbalance at best level."""
        total = self.bid_volumes[0] + self.ask_volumes[0]
        if total == 0:
            return 0.0
        return (self.bid_volumes[0] - self.ask_volumes[0]) / total

    @property
    def depth_imbalance(self) -> float:
        """Cumulative depth imbalance across all levels."""
        bid_tot = self.bid_volumes.sum()
        ask_tot = self.ask_volumes.sum()
        total = bid_tot + ask_tot
        if total == 0:
            return 0.0
        return (bid_tot - ask_tot) / total

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Flatten snapshot to 1-D float32 tensor."""
        features = np.concatenate([
            [self.mid_price / PRICE_NORM_SCALE],
            [self.spread / DEFAULT_TICK_SIZE],
            [self.imbalance],
            [self.depth_imbalance],
            self.bid_prices / PRICE_NORM_SCALE,
            self.bid_volumes / VOL_NORM_SCALE,
            self.ask_prices / PRICE_NORM_SCALE,
            self.ask_volumes / VOL_NORM_SCALE,
            [self.last_trade_price / PRICE_NORM_SCALE],
            [self.last_trade_volume / VOL_NORM_SCALE],
            [self.vwap / PRICE_NORM_SCALE],
            [self.volatility_est],
        ])
        return torch.tensor(features, dtype=torch.float32, device=device)


@dataclasses.dataclass
class AgentOrder:
    """An order submitted by an agent."""
    agent_id: str
    order_type: OrderType
    side: OrderSide
    price: float
    size: float
    timestamp: float = 0.0
    order_id: int = -1

    def is_aggressive(self, snapshot: LOBSnapshot) -> bool:
        if self.order_type == OrderType.MARKET:
            return True
        if self.side == OrderSide.BUY and self.price >= snapshot.best_ask:
            return True
        if self.side == OrderSide.SELL and self.price <= snapshot.best_bid:
            return True
        return False


@dataclasses.dataclass
class Fill:
    """Execution fill report."""
    agent_id: str
    order_id: int
    side: OrderSide
    fill_price: float
    fill_size: float
    timestamp: float
    slippage: float = 0.0
    market_impact: float = 0.0


@dataclasses.dataclass
class AgentState:
    """Per-agent mutable state tracked by the environment."""
    agent_id: str
    cash: float = 0.0
    inventory: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    num_trades: int = 0
    total_volume_traded: float = 0.0
    last_trade_price: float = 0.0
    open_orders: Dict[int, AgentOrder] = dataclasses.field(default_factory=dict)
    fill_history: List[Fill] = dataclasses.field(default_factory=list)
    reward_history: List[float] = dataclasses.field(default_factory=list)
    inventory_history: List[float] = dataclasses.field(default_factory=list)
    pnl_history: List[float] = dataclasses.field(default_factory=list)

    def mark_to_market(self, mid_price: float) -> float:
        self.unrealized_pnl = self.inventory * mid_price + self.cash
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        return self.total_pnl

    def apply_fill(self, fill: Fill) -> None:
        sign = 1.0 if fill.side == OrderSide.BUY else -1.0
        self.inventory += sign * fill.fill_size
        self.cash -= sign * fill.fill_price * fill.fill_size
        self.num_trades += 1
        self.total_volume_traded += fill.fill_size
        self.last_trade_price = fill.fill_price
        self.fill_history.append(fill)


# ---------------------------------------------------------------------------
# Chronos CSV parser
# ---------------------------------------------------------------------------

class ChronosCSVParser:
    """
    Parses Chronos simulator CSV output into LOBSnapshot objects.

    Expected CSV columns (flexible ordering):
        timestamp, mid_price, best_bid, best_ask,
        bid_price_{i}, bid_vol_{i}, ask_price_{i}, ask_vol_{i}  (i=0..depth-1)
        last_trade_price, last_trade_vol, cum_volume, vwap
    """

    REQUIRED_COLS = {"timestamp", "mid_price", "best_bid", "best_ask"}

    def __init__(self, depth: int = DEFAULT_LOB_DEPTH):
        self.depth = depth

    def _make_synthetic_data(self, n_ticks: int = 5000,
                              seed: int = 42) -> pd.DataFrame:
        """Generate synthetic LOB data when no real CSV is available."""
        rng = np.random.default_rng(seed)
        timestamps = np.arange(n_ticks, dtype=float)
        log_returns = rng.normal(0, 0.0002, n_ticks)
        prices = 100.0 * np.exp(np.cumsum(log_returns))

        rows = []
        for t in range(n_ticks):
            mid = prices[t]
            spread = rng.uniform(0.01, 0.05)
            best_bid = mid - spread / 2
            best_ask = mid + spread / 2

            row: Dict[str, Any] = {
                "timestamp": timestamps[t],
                "mid_price": mid,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "last_trade_price": mid + rng.normal(0, spread / 4),
                "last_trade_vol": rng.exponential(5),
                "cum_volume": (t + 1) * rng.exponential(10),
                "vwap": mid + rng.normal(0, 0.001),
            }

            for i in range(self.depth):
                tick_offset = (i + 1) * 0.01
                bid_p = best_bid - i * tick_offset
                ask_p = best_ask + i * tick_offset
                row[f"bid_price_{i}"] = bid_p
                row[f"bid_vol_{i}"] = max(1, rng.exponential(10) * (1 / (i + 1)))
                row[f"ask_price_{i}"] = ask_p
                row[f"ask_vol_{i}"] = max(1, rng.exponential(10) * (1 / (i + 1)))
            rows.append(row)

        return pd.DataFrame(rows)

    def parse_file(self, csv_path: Optional[str] = None,
                   n_synthetic: int = 5000,
                   seed: int = 42) -> List[LOBSnapshot]:
        """Load and parse a Chronos CSV file, or generate synthetic data."""
        if csv_path is not None and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            if csv_path is not None:
                logger.warning("CSV path %s not found; generating synthetic data.", csv_path)
            df = self._make_synthetic_data(n_synthetic, seed)

        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        snapshots: List[LOBSnapshot] = []
        for _, row in df.iterrows():
            snap = self._row_to_snapshot(row)
            if snap is not None:
                snapshots.append(snap)
        logger.info("Parsed %d LOB snapshots.", len(snapshots))
        return snapshots

    def _row_to_snapshot(self, row: pd.Series) -> Optional[LOBSnapshot]:
        try:
            depth = self.depth
            bid_prices = np.array([row.get(f"bid_price_{i}", row["best_bid"] - i * 0.01)
                                   for i in range(depth)], dtype=np.float32)
            bid_volumes = np.array([row.get(f"bid_vol_{i}", max(0.0, 10.0 - i))
                                    for i in range(depth)], dtype=np.float32)
            ask_prices = np.array([row.get(f"ask_price_{i}", row["best_ask"] + i * 0.01)
                                   for i in range(depth)], dtype=np.float32)
            ask_volumes = np.array([row.get(f"ask_vol_{i}", max(0.0, 10.0 - i))
                                    for i in range(depth)], dtype=np.float32)

            # Estimate short-term volatility from spread proxy
            spread = float(row["best_ask"]) - float(row["best_bid"])
            vol_est = spread / max(float(row["mid_price"]), 1e-9)

            return LOBSnapshot(
                timestamp=float(row["timestamp"]),
                mid_price=float(row["mid_price"]),
                best_bid=float(row["best_bid"]),
                best_ask=float(row["best_ask"]),
                spread=spread,
                bid_prices=bid_prices,
                bid_volumes=bid_volumes,
                ask_prices=ask_prices,
                ask_volumes=ask_volumes,
                last_trade_price=float(row.get("last_trade_price", row["mid_price"])),
                last_trade_volume=float(row.get("last_trade_vol", 0.0)),
                cumulative_volume=float(row.get("cum_volume", 0.0)),
                vwap=float(row.get("vwap", row["mid_price"])),
                volatility_est=vol_est,
            )
        except Exception as exc:
            logger.debug("Skipping malformed row: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------

class SimpleExecutionEngine:
    """
    Simulates order matching against the LOB snapshot.
    Supports limit orders (resting), market orders (immediate fill),
    and cancel requests.
    """

    def __init__(self, tick_size: float = DEFAULT_TICK_SIZE,
                 market_impact_factor: float = 1e-4,
                 slippage_factor: float = 5e-5):
        self.tick_size = tick_size
        self.market_impact_factor = market_impact_factor
        self.slippage_factor = slippage_factor
        self._next_order_id = 0

    def _new_order_id(self) -> int:
        oid = self._next_order_id
        self._next_order_id += 1
        return oid

    def submit_order(self, order: AgentOrder,
                     snapshot: LOBSnapshot,
                     agent_state: AgentState) -> Optional[Fill]:
        order.order_id = self._new_order_id()
        order.timestamp = snapshot.timestamp

        if order.order_type == OrderType.NOP:
            return None

        if order.order_type == OrderType.CANCEL:
            self._cancel_order(order, agent_state)
            return None

        if order.order_type == OrderType.MARKET:
            return self._execute_market_order(order, snapshot, agent_state)

        if order.order_type == OrderType.LIMIT:
            return self._execute_limit_order(order, snapshot, agent_state)

        return None

    def _cancel_order(self, order: AgentOrder, agent_state: AgentState) -> None:
        agent_state.open_orders.clear()

    def _execute_market_order(self, order: AgentOrder,
                               snapshot: LOBSnapshot,
                               agent_state: AgentState) -> Optional[Fill]:
        if order.side == OrderSide.BUY:
            base_price = snapshot.best_ask
        else:
            base_price = snapshot.best_bid

        slippage = self._compute_slippage(order.size, snapshot)
        impact = self._compute_market_impact(order.size, snapshot)

        fill_price = base_price
        if order.side == OrderSide.BUY:
            fill_price += slippage + impact
        else:
            fill_price -= slippage + impact

        fill_price = max(fill_price, self.tick_size)

        fill = Fill(
            agent_id=order.agent_id,
            order_id=order.order_id,
            side=order.side,
            fill_price=fill_price,
            fill_size=order.size,
            timestamp=order.timestamp,
            slippage=slippage,
            market_impact=impact,
        )
        agent_state.apply_fill(fill)
        return fill

    def _execute_limit_order(self, order: AgentOrder,
                              snapshot: LOBSnapshot,
                              agent_state: AgentState) -> Optional[Fill]:
        """Execute immediately if aggressive; else rest in book."""
        if order.is_aggressive(snapshot):
            # Treat as market order
            return self._execute_market_order(order, snapshot, agent_state)

        # Passive: add to open orders
        agent_state.open_orders[order.order_id] = order
        return None

    def process_resting_orders(self, snapshot: LOBSnapshot,
                                agent_state: AgentState) -> List[Fill]:
        """Check if any resting limit orders got filled at this tick."""
        fills = []
        to_remove = []
        for oid, order in list(agent_state.open_orders.items()):
            filled = False
            if order.side == OrderSide.BUY and snapshot.best_ask <= order.price:
                fill_price = min(order.price, snapshot.best_ask)
                filled = True
            elif order.side == OrderSide.SELL and snapshot.best_bid >= order.price:
                fill_price = max(order.price, snapshot.best_bid)
                filled = True

            if filled:
                slippage = abs(fill_price - (
                    snapshot.best_ask if order.side == OrderSide.BUY
                    else snapshot.best_bid))
                fill = Fill(
                    agent_id=order.agent_id,
                    order_id=oid,
                    side=order.side,
                    fill_price=fill_price,
                    fill_size=order.size,
                    timestamp=snapshot.timestamp,
                    slippage=slippage,
                )
                agent_state.apply_fill(fill)
                fills.append(fill)
                to_remove.append(oid)

        for oid in to_remove:
            del agent_state.open_orders[oid]

        return fills

    def _compute_slippage(self, size: float, snapshot: LOBSnapshot) -> float:
        """Kyle lambda-style slippage model."""
        depth_vol = snapshot.bid_volumes.sum() + snapshot.ask_volumes.sum()
        if depth_vol <= 0:
            return self.slippage_factor * size
        return self.slippage_factor * size / max(depth_vol, 1.0) * snapshot.mid_price

    def _compute_market_impact(self, size: float, snapshot: LOBSnapshot) -> float:
        """Almgren-Chriss style permanent impact."""
        return self.market_impact_factor * size * snapshot.volatility_est * snapshot.mid_price


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

class ObservationBuilder:
    """Constructs agent observation vectors from LOBSnapshot + AgentState."""

    def __init__(self,
                 depth: int = DEFAULT_LOB_DEPTH,
                 history_len: int = 20,
                 include_agent_state: bool = True):
        self.depth = depth
        self.history_len = history_len
        self.include_agent_state = include_agent_state
        self._price_history: deque = deque(maxlen=history_len)

    @property
    def obs_dim(self) -> int:
        lob_features = 4 + 4 * self.depth   # scalar + bid/ask price+vol
        history_features = self.history_len  # normalised mid-price returns
        agent_features = 6 if self.include_agent_state else 0
        return lob_features + history_features + agent_features

    def build(self,
              snapshot: LOBSnapshot,
              agent_state: AgentState,
              episode_progress: float = 0.0) -> np.ndarray:
        """Return flat observation array."""
        self._price_history.append(snapshot.mid_price)

        # LOB features
        lob_scalar = np.array([
            snapshot.mid_price / PRICE_NORM_SCALE,
            snapshot.spread / DEFAULT_TICK_SIZE,
            snapshot.imbalance,
            snapshot.depth_imbalance,
        ], dtype=np.float32)

        lob_book = np.concatenate([
            snapshot.bid_prices / PRICE_NORM_SCALE,
            snapshot.bid_volumes / VOL_NORM_SCALE,
            snapshot.ask_prices / PRICE_NORM_SCALE,
            snapshot.ask_volumes / VOL_NORM_SCALE,
        ]).astype(np.float32)

        # Price return history
        history = np.zeros(self.history_len, dtype=np.float32)
        prices = list(self._price_history)
        if len(prices) > 1:
            for i in range(1, len(prices)):
                ret = (prices[i] - prices[i - 1]) / max(prices[i - 1], 1e-9)
                history[i - 1] = np.clip(ret, -0.1, 0.1)

        parts = [lob_scalar, lob_book, history]

        if self.include_agent_state:
            pnl = agent_state.mark_to_market(snapshot.mid_price)
            agent_obs = np.array([
                np.clip(agent_state.inventory / DEFAULT_MAX_INVENTORY, -1.0, 1.0),
                np.clip(pnl / 1000.0, -10.0, 10.0),
                np.clip(agent_state.num_trades / 100.0, 0.0, 1.0),
                episode_progress,
                snapshot.vwap / PRICE_NORM_SCALE,
                snapshot.volatility_est,
            ], dtype=np.float32)
            parts.append(agent_obs)

        return np.concatenate(parts)

    def reset(self) -> None:
        self._price_history.clear()


# ---------------------------------------------------------------------------
# Action decoder
# ---------------------------------------------------------------------------

class ActionDecoder:
    """
    Decodes discrete + continuous action tuples to AgentOrder objects.

    Action space (MultiDiscrete + Box hybrid):
      - Discrete: order_type (4), side (2)  -> encoded as single int [0..7]
      - Continuous: price_offset (normalised), size (normalised)
    """

    def __init__(self,
                 tick_size: float = DEFAULT_TICK_SIZE,
                 max_order_size: int = DEFAULT_MAX_ORDER_SIZE,
                 price_offset_ticks: int = 10):
        self.tick_size = tick_size
        self.max_order_size = max_order_size
        self.price_offset_ticks = price_offset_ticks

    @property
    def action_space(self) -> spaces.Dict:
        return spaces.Dict({
            "order_type_side": spaces.Discrete(8),   # (type, side) combo
            "price_offset": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "size": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

    @property
    def flat_action_space(self) -> spaces.Box:
        """Flat continuous action: [type_side_one_hot(8), price_offset, size]."""
        return spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)

    def decode(self, action: Union[Dict, np.ndarray],
               snapshot: LOBSnapshot,
               agent_id: str) -> AgentOrder:
        if isinstance(action, np.ndarray):
            return self._decode_flat(action, snapshot, agent_id)
        return self._decode_dict(action, snapshot, agent_id)

    def _decode_flat(self, action: np.ndarray,
                     snapshot: LOBSnapshot,
                     agent_id: str) -> AgentOrder:
        """Decode flat 10-d continuous action vector."""
        action = np.clip(action, -1.0, 1.0)
        # First 4 dims encode order type logits
        type_logits = action[:4]
        order_type = OrderType(int(np.argmax(type_logits)))
        # Dim 4 encodes side
        side = OrderSide.BUY if action[4] > 0 else OrderSide.SELL
        # Dim 5: price offset (normalised to ±price_offset_ticks ticks)
        price_offset_raw = float(action[5])
        price_offset = price_offset_raw * self.price_offset_ticks * self.tick_size
        # Dim 6: size
        size_raw = (float(action[6]) + 1.0) / 2.0  # [0,1]
        size = max(DEFAULT_LOT_SIZE,
                   round(size_raw * self.max_order_size / DEFAULT_LOT_SIZE) * DEFAULT_LOT_SIZE)

        if side == OrderSide.BUY:
            price = snapshot.best_bid + price_offset
        else:
            price = snapshot.best_ask - price_offset

        price = max(price, self.tick_size)

        return AgentOrder(
            agent_id=agent_id,
            order_type=order_type,
            side=side,
            price=round(price / self.tick_size) * self.tick_size,
            size=size,
        )

    def _decode_dict(self, action: Dict,
                     snapshot: LOBSnapshot,
                     agent_id: str) -> AgentOrder:
        combo = int(action["order_type_side"]) % 8
        order_type = OrderType(combo // 2)
        side = OrderSide(combo % 2)

        price_offset_raw = float(np.clip(action["price_offset"], -1.0, 1.0))
        price_offset = price_offset_raw * self.price_offset_ticks * self.tick_size

        size_raw = float(np.clip(action["size"], 0.0, 1.0))
        size = max(DEFAULT_LOT_SIZE, round(size_raw * self.max_order_size))

        if side == OrderSide.BUY:
            price = snapshot.best_bid + price_offset
        else:
            price = snapshot.best_ask - price_offset

        return AgentOrder(
            agent_id=agent_id,
            order_type=order_type,
            side=side,
            price=max(price, self.tick_size),
            size=size,
        )


# ---------------------------------------------------------------------------
# Reward computer
# ---------------------------------------------------------------------------

class RewardComputer:
    """
    Computes per-step reward for an agent from state transitions.
    Supports multiple reward modes and mixing.
    """

    def __init__(self,
                 mode: RewardMode = RewardMode.MARK_TO_MARKET,
                 inventory_penalty_coef: float = 0.01,
                 slippage_penalty_coef: float = 0.1,
                 sharpe_window: int = 50,
                 clip_range: float = 10.0):
        self.mode = mode
        self.inventory_penalty_coef = inventory_penalty_coef
        self.slippage_penalty_coef = slippage_penalty_coef
        self.sharpe_window = sharpe_window
        self.clip_range = clip_range
        self._pnl_buffer: deque = deque(maxlen=sharpe_window)

    def compute(self,
                agent_state: AgentState,
                snapshot_prev: LOBSnapshot,
                snapshot_curr: LOBSnapshot,
                fills: List[Fill]) -> float:
        if self.mode == RewardMode.REALIZED_PNL:
            reward = self._realized_pnl_reward(agent_state, fills)
        elif self.mode == RewardMode.MARK_TO_MARKET:
            reward = self._mtm_reward(agent_state, snapshot_prev, snapshot_curr)
        elif self.mode == RewardMode.SHAPED_SHARPE:
            reward = self._sharpe_reward(agent_state, snapshot_curr)
        elif self.mode == RewardMode.ADVERSARIAL:
            reward = self._adversarial_reward(agent_state, snapshot_curr)
        else:
            reward = 0.0

        reward -= self._inventory_penalty(agent_state)
        reward -= self._slippage_penalty(fills)
        reward = float(np.clip(reward, -self.clip_range, self.clip_range))
        agent_state.reward_history.append(reward)
        return reward

    def _realized_pnl_reward(self, agent_state: AgentState,
                              fills: List[Fill]) -> float:
        total = 0.0
        for fill in fills:
            if fill.side == OrderSide.SELL:
                if agent_state.last_trade_price > 0:
                    total += (fill.fill_price - agent_state.last_trade_price) * fill.fill_size
        return total

    def _mtm_reward(self, agent_state: AgentState,
                    snap_prev: LOBSnapshot,
                    snap_curr: LOBSnapshot) -> float:
        pnl_prev = agent_state.inventory * snap_prev.mid_price + agent_state.cash
        pnl_curr = agent_state.inventory * snap_curr.mid_price + agent_state.cash
        return pnl_curr - pnl_prev

    def _sharpe_reward(self, agent_state: AgentState,
                       snap: LOBSnapshot) -> float:
        pnl = agent_state.mark_to_market(snap.mid_price)
        self._pnl_buffer.append(pnl)
        if len(self._pnl_buffer) < 3:
            return 0.0
        pnl_arr = np.array(self._pnl_buffer)
        returns = np.diff(pnl_arr)
        std = returns.std() + 1e-9
        return float(returns[-1] / std)

    def _adversarial_reward(self, agent_state: AgentState,
                             snap: LOBSnapshot) -> float:
        """Reward that incentivises destabilising the market."""
        return abs(snap.spread) * 10.0 - abs(agent_state.inventory) * 0.001

    def _inventory_penalty(self, agent_state: AgentState) -> float:
        inv_ratio = abs(agent_state.inventory) / DEFAULT_MAX_INVENTORY
        return self.inventory_penalty_coef * (inv_ratio ** 2)

    def _slippage_penalty(self, fills: List[Fill]) -> float:
        return sum(f.slippage for f in fills) * self.slippage_penalty_coef

    def reset(self) -> None:
        self._pnl_buffer.clear()


# ---------------------------------------------------------------------------
# Episode manager (scenario selection)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class EpisodeConfig:
    """Configuration for a single episode."""
    scenario_name: str = "normal"
    start_idx: int = 0
    end_idx: int = DEFAULT_EPISODE_LEN
    seed: int = 0
    difficulty: float = 0.5
    # Perturbations
    flash_crash_enabled: bool = False
    flash_crash_tick: int = -1
    flash_crash_magnitude: float = 0.05
    vol_multiplier: float = 1.0
    spread_multiplier: float = 1.0


class EpisodeManager:
    """Manages episode lifecycle and scenario selection."""

    SCENARIOS = [
        "normal",
        "high_vol",
        "low_liquidity",
        "trending",
        "mean_reverting",
        "flash_crash",
    ]

    def __init__(self,
                 snapshots: List[LOBSnapshot],
                 episode_len: int = DEFAULT_EPISODE_LEN,
                 seed: int = 0):
        self.snapshots = snapshots
        self.episode_len = episode_len
        self.rng = np.random.default_rng(seed)
        self._episode_count = 0

    def sample_episode(self, scenario: Optional[str] = None,
                       difficulty: float = 0.5) -> EpisodeConfig:
        if scenario is None:
            scenario = self.rng.choice(self.SCENARIOS)

        max_start = max(0, len(self.snapshots) - self.episode_len - 1)
        start_idx = int(self.rng.integers(0, max(1, max_start)))
        end_idx = start_idx + self.episode_len

        cfg = EpisodeConfig(
            scenario_name=scenario,
            start_idx=start_idx,
            end_idx=min(end_idx, len(self.snapshots) - 1),
            seed=int(self.rng.integers(0, 2 ** 31)),
            difficulty=difficulty,
        )

        if scenario == "high_vol":
            cfg.vol_multiplier = 1.0 + 2.0 * difficulty
            cfg.spread_multiplier = 1.0 + difficulty
        elif scenario == "low_liquidity":
            cfg.spread_multiplier = 1.0 + 3.0 * difficulty
        elif scenario == "flash_crash":
            cfg.flash_crash_enabled = True
            crash_offset = int(self.rng.integers(100, max(101, self.episode_len - 100)))
            cfg.flash_crash_tick = start_idx + crash_offset
            cfg.flash_crash_magnitude = 0.03 + 0.07 * difficulty

        self._episode_count += 1
        return cfg

    def get_snapshots_for_episode(self, cfg: EpisodeConfig) -> List[LOBSnapshot]:
        snaps = copy.deepcopy(self.snapshots[cfg.start_idx:cfg.end_idx])
        # Apply perturbations
        for i, snap in enumerate(snaps):
            snap.spread *= cfg.spread_multiplier
            snap.volatility_est *= cfg.vol_multiplier
            snap.best_bid = snap.mid_price - snap.spread / 2
            snap.best_ask = snap.mid_price + snap.spread / 2

            abs_tick = cfg.start_idx + i
            if cfg.flash_crash_enabled and abs_tick >= cfg.flash_crash_tick:
                decay = max(0.0, 1.0 - (abs_tick - cfg.flash_crash_tick) / 200.0)
                drop = cfg.flash_crash_magnitude * decay
                snap.mid_price *= (1.0 - drop)
                snap.best_bid = snap.mid_price - snap.spread / 2
                snap.best_ask = snap.mid_price + snap.spread / 2
                snap.bid_prices = np.linspace(snap.best_bid,
                                              snap.best_bid - snap.spread * 5,
                                              len(snap.bid_prices)).astype(np.float32)
                snap.ask_prices = np.linspace(snap.best_ask,
                                              snap.best_ask + snap.spread * 5,
                                              len(snap.ask_prices)).astype(np.float32)
        return snaps


# ---------------------------------------------------------------------------
# Core Single-Agent Gymnasium Environment
# ---------------------------------------------------------------------------

class ChronosLOBEnv(gym.Env):
    """
    Gymnasium-compatible single-agent LOB trading environment backed by
    Chronos simulator data.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}

    def __init__(self,
                 csv_path: Optional[str] = None,
                 depth: int = DEFAULT_LOB_DEPTH,
                 episode_len: int = DEFAULT_EPISODE_LEN,
                 reward_mode: RewardMode = RewardMode.MARK_TO_MARKET,
                 inventory_penalty_coef: float = 0.01,
                 max_inventory: float = DEFAULT_MAX_INVENTORY,
                 tick_size: float = DEFAULT_TICK_SIZE,
                 max_order_size: int = DEFAULT_MAX_ORDER_SIZE,
                 history_len: int = 20,
                 seed: int = 42,
                 scenario: Optional[str] = None,
                 render_mode: Optional[str] = None,
                 n_synthetic: int = 5000):
        super().__init__()

        self.depth = depth
        self.episode_len = episode_len
        self.reward_mode = reward_mode
        self.max_inventory = max_inventory
        self.tick_size = tick_size
        self.max_order_size = max_order_size
        self.scenario = scenario
        self.render_mode = render_mode

        # Components
        parser = ChronosCSVParser(depth=depth)
        self._all_snapshots = parser.parse_file(csv_path, n_synthetic=n_synthetic, seed=seed)

        self._episode_manager = EpisodeManager(self._all_snapshots, episode_len, seed)
        self._obs_builder = ObservationBuilder(depth, history_len)
        self._action_decoder = ActionDecoder(tick_size, max_order_size)
        self._exec_engine = SimpleExecutionEngine(tick_size)
        self._reward_computer = RewardComputer(
            mode=reward_mode,
            inventory_penalty_coef=inventory_penalty_coef,
        )

        # Episode state
        self._episode_snapshots: List[LOBSnapshot] = []
        self._tick: int = 0
        self._agent_state: Optional[AgentState] = None
        self._episode_cfg: Optional[EpisodeConfig] = None

        # Spaces
        obs_dim = self._obs_builder.obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = self._action_decoder.flat_action_space

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._episode_manager.rng = np.random.default_rng(seed)

        difficulty = (options or {}).get("difficulty", 0.5)
        scenario = (options or {}).get("scenario", self.scenario)

        self._episode_cfg = self._episode_manager.sample_episode(scenario, difficulty)
        self._episode_snapshots = self._episode_manager.get_snapshots_for_episode(
            self._episode_cfg
        )
        self._tick = 0
        self._agent_state = AgentState(agent_id="agent_0")
        self._obs_builder.reset()
        self._reward_computer.reset()

        obs = self._obs_builder.build(
            self._episode_snapshots[0],
            self._agent_state,
            episode_progress=0.0,
        )
        return obs, self._info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self._agent_state is not None, "Call reset() first."
        snap_prev = self._episode_snapshots[self._tick]

        # Decode & submit action
        order = self._action_decoder.decode(action, snap_prev, "agent_0")
        fill = self._exec_engine.submit_order(order, snap_prev, self._agent_state)

        # Advance tick
        self._tick += 1
        done = self._tick >= len(self._episode_snapshots) - 1
        snap_curr = self._episode_snapshots[min(self._tick, len(self._episode_snapshots) - 1)]

        # Process resting orders
        fills = self._exec_engine.process_resting_orders(snap_curr, self._agent_state)
        if fill is not None:
            fills = [fill] + fills

        # Compute reward
        reward = self._reward_computer.compute(
            self._agent_state, snap_prev, snap_curr, fills
        )

        # Inventory limit check
        truncated = abs(self._agent_state.inventory) > self.max_inventory

        # Observation
        ep_progress = self._tick / max(len(self._episode_snapshots), 1)
        obs = self._obs_builder.build(snap_curr, self._agent_state, ep_progress)

        return obs, reward, done, truncated, self._info(fills=fills)

    def render(self) -> Optional[str]:
        if not self._episode_snapshots or self._agent_state is None:
            return None
        snap = self._episode_snapshots[min(self._tick, len(self._episode_snapshots) - 1)]
        msg = (
            f"Tick {self._tick:5d} | Mid={snap.mid_price:.4f} | "
            f"Spread={snap.spread:.4f} | "
            f"Inv={self._agent_state.inventory:+.1f} | "
            f"PnL={self._agent_state.total_pnl:+.2f}"
        )
        if self.render_mode == "human":
            print(msg)
        return msg

    def close(self) -> None:
        pass

    def _info(self, fills: Optional[List[Fill]] = None) -> Dict:
        state = self._agent_state
        snap = (self._episode_snapshots[min(self._tick, len(self._episode_snapshots) - 1)]
                if self._episode_snapshots else None)
        return {
            "inventory": state.inventory if state else 0.0,
            "realized_pnl": state.realized_pnl if state else 0.0,
            "total_pnl": state.total_pnl if state else 0.0,
            "num_trades": state.num_trades if state else 0,
            "tick": self._tick,
            "mid_price": snap.mid_price if snap else 0.0,
            "spread": snap.spread if snap else 0.0,
            "fills": fills or [],
        }


# ---------------------------------------------------------------------------
# Multi-Agent PettingZoo Parallel Environment
# ---------------------------------------------------------------------------

class ChronosMARLEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv wrapping the Chronos LOB simulator.
    All agents act simultaneously each step.
    """

    metadata = {"render_modes": ["human", "ansi"], "name": "chronos_marl_v1"}

    def __init__(self,
                 n_agents: int = 4,
                 csv_path: Optional[str] = None,
                 depth: int = DEFAULT_LOB_DEPTH,
                 episode_len: int = DEFAULT_EPISODE_LEN,
                 reward_mode: RewardMode = RewardMode.MARK_TO_MARKET,
                 inventory_penalty_coef: float = 0.01,
                 agent_reward_mixing: Optional[Dict[str, float]] = None,
                 cooperative_coef: float = 0.1,
                 seed: int = 42,
                 n_synthetic: int = 5000):
        super().__init__()

        self.n_agents_total = n_agents
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agents = list(self.possible_agents)
        self.cooperative_coef = cooperative_coef
        self.agent_reward_mixing = agent_reward_mixing or {}

        parser = ChronosCSVParser(depth=depth)
        self._all_snapshots = parser.parse_file(csv_path, n_synthetic=n_synthetic, seed=seed)

        self._episode_manager = EpisodeManager(self._all_snapshots, episode_len, seed)
        self._exec_engine = SimpleExecutionEngine()
        self._action_decoder = ActionDecoder()

        self._obs_builders: Dict[str, ObservationBuilder] = {
            aid: ObservationBuilder(depth) for aid in self.possible_agents
        }
        self._reward_computers: Dict[str, RewardComputer] = {
            aid: RewardComputer(mode=reward_mode,
                                inventory_penalty_coef=inventory_penalty_coef)
            for aid in self.possible_agents
        }

        self._episode_snapshots: List[LOBSnapshot] = []
        self._tick: int = 0
        self._agent_states: Dict[str, AgentState] = {}
        self._episode_cfg: Optional[EpisodeConfig] = None

        obs_dim = self._obs_builders["agent_0"].obs_dim
        self.observation_spaces = {
            aid: spaces.Box(low=-np.inf, high=np.inf,
                            shape=(obs_dim,), dtype=np.float32)
            for aid in self.possible_agents
        }
        self.action_spaces = {
            aid: self._action_decoder.flat_action_space
            for aid in self.possible_agents
        }

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        if seed is not None:
            self._episode_manager.rng = np.random.default_rng(seed)

        difficulty = (options or {}).get("difficulty", 0.5)
        scenario = (options or {}).get("scenario", None)

        self._episode_cfg = self._episode_manager.sample_episode(scenario, difficulty)
        self._episode_snapshots = self._episode_manager.get_snapshots_for_episode(
            self._episode_cfg
        )
        self._tick = 0
        self.agents = list(self.possible_agents)
        self._agent_states = {
            aid: AgentState(agent_id=aid) for aid in self.possible_agents
        }
        for builder in self._obs_builders.values():
            builder.reset()
        for rc in self._reward_computers.values():
            rc.reset()

        snap = self._episode_snapshots[0]
        obs = {
            aid: self._obs_builders[aid].build(snap, self._agent_states[aid], 0.0)
            for aid in self.agents
        }
        infos = {aid: {} for aid in self.agents}
        return obs, infos

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict, Dict, Dict, Dict, Dict
    ]:
        if self._tick >= len(self._episode_snapshots) - 1:
            # Terminal
            obs = {aid: np.zeros(self._obs_builders[aid].obs_dim, dtype=np.float32)
                   for aid in self.agents}
            rewards = {aid: 0.0 for aid in self.agents}
            terms = {aid: True for aid in self.agents}
            truncs = {aid: False for aid in self.agents}
            infos = {aid: {} for aid in self.agents}
            self.agents = []
            return obs, rewards, terms, truncs, infos

        snap_prev = self._episode_snapshots[self._tick]

        # All agents submit orders
        agent_fills: Dict[str, List[Fill]] = {aid: [] for aid in self.agents}
        for aid, action in actions.items():
            if aid not in self.agents:
                continue
            order = self._action_decoder.decode(action, snap_prev, aid)
            fill = self._exec_engine.submit_order(order, snap_prev,
                                                   self._agent_states[aid])
            if fill is not None:
                agent_fills[aid].append(fill)

        self._tick += 1
        snap_curr = self._episode_snapshots[min(self._tick,
                                                 len(self._episode_snapshots) - 1)]

        # Process resting orders
        for aid in self.agents:
            resting_fills = self._exec_engine.process_resting_orders(
                snap_curr, self._agent_states[aid]
            )
            agent_fills[aid].extend(resting_fills)

        # Compute per-agent rewards
        ep_progress = self._tick / max(len(self._episode_snapshots), 1)
        rewards: Dict[str, float] = {}
        obs: Dict[str, np.ndarray] = {}
        terminations: Dict[str, bool] = {}
        truncations: Dict[str, bool] = {}
        infos: Dict[str, Dict] = {}

        done = self._tick >= len(self._episode_snapshots) - 1

        for aid in self.agents:
            r = self._reward_computers[aid].compute(
                self._agent_states[aid], snap_prev, snap_curr, agent_fills[aid]
            )
            rewards[aid] = r
            obs[aid] = self._obs_builders[aid].build(
                snap_curr, self._agent_states[aid], ep_progress
            )
            terminations[aid] = done
            truncations[aid] = abs(self._agent_states[aid].inventory) > DEFAULT_MAX_INVENTORY
            infos[aid] = {
                "inventory": self._agent_states[aid].inventory,
                "total_pnl": self._agent_states[aid].total_pnl,
                "fills": agent_fills[aid],
            }

        # Cooperative reward mixing
        if self.cooperative_coef > 0:
            mean_r = np.mean(list(rewards.values()))
            for aid in self.agents:
                rewards[aid] = (1 - self.cooperative_coef) * rewards[aid] + \
                               self.cooperative_coef * mean_r

        if done:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def render(self) -> None:
        if not self._episode_snapshots:
            return
        snap = self._episode_snapshots[min(self._tick, len(self._episode_snapshots) - 1)]
        pnls = {aid: self._agent_states[aid].total_pnl
                for aid in self.possible_agents
                if aid in self._agent_states}
        print(f"Tick {self._tick} | Mid={snap.mid_price:.4f} | "
              f"Spread={snap.spread:.4f} | PnLs={pnls}")


# ---------------------------------------------------------------------------
# Vectorised Environment (VecEnv)
# ---------------------------------------------------------------------------

class VecChronosEnv:
    """
    Synchronous vectorised wrapper over N independent ChronosLOBEnv instances.
    Provides batch_step / batch_reset for data-parallel training.
    """

    def __init__(self,
                 n_envs: int = 8,
                 env_kwargs: Optional[Dict] = None):
        env_kwargs = env_kwargs or {}
        self.n_envs = n_envs
        self.envs: List[ChronosLOBEnv] = [
            ChronosLOBEnv(seed=i, **env_kwargs) for i in range(n_envs)
        ]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self, options: Optional[List[Optional[Dict]]] = None
              ) -> Tuple[np.ndarray, List[Dict]]:
        options = options or [None] * self.n_envs
        results = [env.reset(options=opt) for env, opt in zip(self.envs, options)]
        obs_list, info_list = zip(*results)
        return np.stack(obs_list), list(info_list)

    def step(self, actions: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]
    ]:
        assert len(actions) == self.n_envs
        obs_list, reward_list, done_list, trunc_list, info_list = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, done, trunc, info = env.step(action)
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            trunc_list.append(trunc)
            info_list.append(info)

            if done or trunc:
                obs_reset, _ = env.reset()
                obs_list[-1] = obs_reset  # auto-reset

        return (
            np.stack(obs_list),
            np.array(reward_list, dtype=np.float32),
            np.array(done_list, dtype=bool),
            np.array(trunc_list, dtype=bool),
            info_list,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()

    @property
    def obs_dim(self) -> int:
        return self.observation_space.shape[0]

    @property
    def act_dim(self) -> int:
        return self.action_space.shape[0]


# ---------------------------------------------------------------------------
# Async VecEnv (multiprocess stub — uses threading for simplicity)
# ---------------------------------------------------------------------------

class AsyncVecChronosEnv(VecChronosEnv):
    """
    Asynchronous vectorised environment using concurrent.futures ThreadPoolExecutor.
    Each environment runs in its own thread for overlapping I/O.
    """

    def __init__(self, n_envs: int = 8, env_kwargs: Optional[Dict] = None,
                 max_workers: Optional[int] = None):
        super().__init__(n_envs, env_kwargs)
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=max_workers or n_envs)

    def step(self, actions: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]
    ]:
        futures = [
            self._pool.submit(env.step, action)
            for env, action in zip(self.envs, actions)
        ]
        results = [f.result() for f in futures]

        obs_list, reward_list, done_list, trunc_list, info_list = [], [], [], [], []
        for i, (obs, reward, done, trunc, info) in enumerate(results):
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            trunc_list.append(trunc)
            info_list.append(info)
            if done or trunc:
                obs_reset, _ = self.envs[i].reset()
                obs_list[-1] = obs_reset

        return (
            np.stack(obs_list),
            np.array(reward_list, dtype=np.float32),
            np.array(done_list, dtype=bool),
            np.array(trunc_list, dtype=bool),
            info_list,
        )

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        super().close()


# ---------------------------------------------------------------------------
# Gymnasium-registration helper
# ---------------------------------------------------------------------------

def register_envs() -> None:
    """Register Chronos envs with Gymnasium registry."""
    try:
        gym.register(
            id="ChronosLOB-v1",
            entry_point="hyper_agent.chronos_env_bridge:ChronosLOBEnv",
            max_episode_steps=DEFAULT_EPISODE_LEN,
        )
        logger.info("Registered ChronosLOB-v1 with Gymnasium.")
    except Exception as exc:
        logger.debug("Could not register env: %s", exc)


# ---------------------------------------------------------------------------
# Environment wrappers (observation / reward normalisers)
# ---------------------------------------------------------------------------

class RunningMeanStdNormaliser:
    """Welford online algorithm for running mean/std normalisation."""

    def __init__(self, shape: Tuple[int, ...], clip: float = 10.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-8
        self.clip = clip

    def update(self, x: np.ndarray) -> None:
        batch_mean = x.mean(axis=0) if x.ndim > 1 else x
        batch_var = x.var(axis=0) if x.ndim > 1 else np.zeros_like(x)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count

    def normalise(self, x: np.ndarray) -> np.ndarray:
        norm = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(norm, -self.clip, self.clip).astype(np.float32)


class NormalisedObsWrapper(gym.Wrapper):
    """Wrapper that normalises observations using running statistics."""

    def __init__(self, env: ChronosLOBEnv):
        super().__init__(env)
        self._normaliser = RunningMeanStdNormaliser(
            (env.observation_space.shape[0],)
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        self._normaliser.update(obs)
        return self._normaliser.normalise(obs), info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, done, trunc, info = self.env.step(action)
        self._normaliser.update(obs)
        return self._normaliser.normalise(obs), reward, done, trunc, info


class FrameStackWrapper(gym.Wrapper):
    """Stack the last N observations."""

    def __init__(self, env: ChronosLOBEnv, n_stack: int = 4):
        super().__init__(env)
        self.n_stack = n_stack
        obs_dim = env.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim * n_stack,), dtype=np.float32,
        )
        self._frames: deque = deque(maxlen=n_stack)

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        self._frames.clear()
        for _ in range(self.n_stack):
            self._frames.append(obs)
        return self._stacked(), info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, done, trunc, info = self.env.step(action)
        self._frames.append(obs)
        return self._stacked(), reward, done, trunc, info

    def _stacked(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=0)


# ---------------------------------------------------------------------------
# Diagnostic / logging utilities
# ---------------------------------------------------------------------------

class EpisodeTracker:
    """Tracks per-episode statistics for monitoring."""

    def __init__(self, window: int = 100):
        self.window = window
        self._episode_returns: deque = deque(maxlen=window)
        self._episode_lengths: deque = deque(maxlen=window)
        self._inventory_peaks: deque = deque(maxlen=window)
        self._n_episodes: int = 0

    def record_episode(self, total_return: float, length: int,
                       peak_inventory: float) -> None:
        self._episode_returns.append(total_return)
        self._episode_lengths.append(length)
        self._inventory_peaks.append(peak_inventory)
        self._n_episodes += 1

    def summary(self) -> Dict[str, float]:
        if not self._episode_returns:
            return {}
        returns = np.array(self._episode_returns)
        return {
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std()),
            "min_return": float(returns.min()),
            "max_return": float(returns.max()),
            "mean_length": float(np.mean(self._episode_lengths)),
            "mean_peak_inventory": float(np.mean(self._inventory_peaks)),
            "n_episodes": self._n_episodes,
            "sharpe": float(returns.mean() / (returns.std() + 1e-9)),
        }


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_vec_env(n_envs: int = 8,
                 async_mode: bool = False,
                 **env_kwargs) -> VecChronosEnv:
    """Convenience factory for vectorised environments."""
    cls = AsyncVecChronosEnv if async_mode else VecChronosEnv
    return cls(n_envs=n_envs, env_kwargs=env_kwargs)


def make_marl_env(n_agents: int = 4, **kwargs) -> ChronosMARLEnv:
    """Convenience factory for multi-agent environment."""
    return ChronosMARLEnv(n_agents=n_agents, **kwargs)


def make_single_env(normalise: bool = True,
                    frame_stack: int = 0,
                    **env_kwargs) -> gym.Env:
    """Convenience factory with optional wrappers."""
    env = ChronosLOBEnv(**env_kwargs)
    if normalise:
        env = NormalisedObsWrapper(env)
    if frame_stack > 1:
        env = FrameStackWrapper(env, n_stack=frame_stack)
    return env


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== ChronosLOBEnv smoke test ===")
    env = ChronosLOBEnv(n_synthetic=2000, episode_len=200, seed=0)
    obs, info = env.reset()
    print(f"Obs shape: {obs.shape}")
    total_reward = 0.0
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        if done or trunc:
            break
    print(f"Total reward: {total_reward:.4f}")

    print("\n=== ChronosMARLEnv smoke test ===")
    marl_env = ChronosMARLEnv(n_agents=3, n_synthetic=2000, episode_len=200)
    obs_dict, _ = marl_env.reset()
    for aid, o in obs_dict.items():
        print(f"  {aid}: obs shape {o.shape}")

    print("\n=== VecChronosEnv smoke test ===")
    vec_env = VecChronosEnv(n_envs=4, env_kwargs={"n_synthetic": 1000, "episode_len": 100})
    batch_obs, _ = vec_env.reset()
    print(f"Batch obs shape: {batch_obs.shape}")
    actions = np.stack([vec_env.action_space.sample() for _ in range(4)])
    batch_obs, batch_r, batch_done, batch_trunc, _ = vec_env.step(actions)
    print(f"Batch reward: {batch_r}")
    vec_env.close()

    print("\nAll smoke tests passed.")
