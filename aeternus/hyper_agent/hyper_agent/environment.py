"""
environment.py — Full multi-asset trading environment for Hyper-Agent MARL.

Implements:
- OrderBook simulation with price-time priority
- Multi-asset support with correlated price dynamics
- Reward shaping: PnL, Sharpe ratio, drawdown penalty, inventory penalty
- Multi-agent observation space (per-agent and global)
- Continuous action space: bid price offset, ask price offset, bid size, ask size
- Done conditions: max steps, bankruptcy, circuit breaker
- Market microstructure: spread, depth, impact, fees
- Information asymmetry and private signals
- Noise traders for background liquidity
- Flash crash injection
"""

from __future__ import annotations

import math
import heapq
import logging
import collections
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUY = 1
SELL = -1
HOLD = 0

ORDER_LIMIT = "limit"
ORDER_MARKET = "market"
ORDER_CANCEL = "cancel"

SIDE_BID = "bid"
SIDE_ASK = "ask"

DEFAULT_TICK_SIZE = 0.01
DEFAULT_LOT_SIZE = 1.0
DEFAULT_MAX_POSITION = 1000.0
DEFAULT_INITIAL_CASH = 100_000.0
DEFAULT_TRANSACTION_COST = 0.0005
DEFAULT_MAX_STEPS = 2000
DEFAULT_NUM_ASSETS = 4
DEFAULT_NUM_AGENTS = 8

EPS = 1e-8


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Order:
    """Single limit or market order."""
    price: float
    timestamp: int
    order_id: int = field(compare=False)
    agent_id: int = field(compare=False)
    asset_id: int = field(compare=False)
    side: str = field(compare=False)
    size: float = field(compare=False)
    order_type: str = field(compare=False, default=ORDER_LIMIT)
    filled: float = field(compare=False, default=0.0)
    active: bool = field(compare=False, default=True)

    @property
    def remaining(self) -> float:
        return max(0.0, self.size - self.filled)

    def __repr__(self) -> str:
        return (
            f"Order(id={self.order_id}, agent={self.agent_id}, "
            f"asset={self.asset_id}, side={self.side}, "
            f"price={self.price:.4f}, size={self.size:.2f}, filled={self.filled:.2f})"
        )


@dataclass
class Trade:
    """A matched trade between two orders."""
    trade_id: int
    asset_id: int
    price: float
    size: float
    aggressor_id: int
    passive_id: int
    aggressor_agent: int
    passive_agent: int
    timestamp: int
    side: str

    def __repr__(self) -> str:
        return (
            f"Trade(id={self.trade_id}, asset={self.asset_id}, "
            f"price={self.price:.4f}, size={self.size:.2f}, ts={self.timestamp})"
        )


@dataclass
class AgentState:
    """Per-agent portfolio state."""
    agent_id: int
    cash: float
    positions: np.ndarray
    avg_cost: np.ndarray
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades: int = 0
    total_volume: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    returns_history: List[float] = field(default_factory=list)
    equity_history: List[float] = field(default_factory=list)

    def net_equity(self, mid_prices: np.ndarray) -> float:
        return self.cash + float(np.dot(self.positions, mid_prices))

    def update_unrealized(self, mid_prices: np.ndarray) -> None:
        self.unrealized_pnl = float(np.dot(self.positions, mid_prices - self.avg_cost))


# ---------------------------------------------------------------------------
# Order Book
# ---------------------------------------------------------------------------

class OrderBook:
    """
    Price-time priority order book for a single asset.

    Bids: max-heap (stored as negative price).
    Asks: min-heap.
    Matching is immediate upon order submission.
    """

    def __init__(self, asset_id: int, tick_size: float = DEFAULT_TICK_SIZE):
        self.asset_id = asset_id
        self.tick_size = tick_size

        self._bids: List[Tuple] = []   # (-price, timestamp, order)
        self._asks: List[Tuple] = []   # (price, timestamp, order)
        self._orders: Dict[int, Order] = {}

        self._next_order_id: int = 0
        self._next_trade_id: int = 0

        self.last_trade_price: float = 0.0
        self.last_trade_size: float = 0.0
        self.volume_today: float = 0.0
        self.trade_count: int = 0
        self._recent_trades: collections.deque = collections.deque(maxlen=200)

    # ---- Public API -------------------------------------------------------

    def submit_limit_order(
        self,
        agent_id: int,
        side: str,
        price: float,
        size: float,
        timestamp: int,
    ) -> Tuple[int, List[Trade]]:
        price = self._round_tick(price)
        size = max(size, 0.0)
        if size < EPS:
            return -1, []

        oid = self._next_order_id
        self._next_order_id += 1
        order = Order(
            price=price, timestamp=timestamp, order_id=oid,
            agent_id=agent_id, asset_id=self.asset_id,
            side=side, size=size, order_type=ORDER_LIMIT,
        )
        self._orders[oid] = order
        trades = self._match_order(order, timestamp)
        if order.active and order.remaining > EPS:
            self._add_to_book(order)
        return oid, trades

    def submit_market_order(
        self,
        agent_id: int,
        side: str,
        size: float,
        timestamp: int,
    ) -> Tuple[int, List[Trade]]:
        size = max(size, 0.0)
        if size < EPS:
            return -1, []
        oid = self._next_order_id
        self._next_order_id += 1
        price = 1e12 if side == SIDE_BID else -1e12
        order = Order(
            price=price, timestamp=timestamp, order_id=oid,
            agent_id=agent_id, asset_id=self.asset_id,
            side=side, size=size, order_type=ORDER_MARKET,
        )
        self._orders[oid] = order
        trades = self._match_order(order, timestamp)
        order.active = False
        return oid, trades

    def cancel_order(self, order_id: int) -> bool:
        order = self._orders.get(order_id)
        if order is None or not order.active:
            return False
        order.active = False
        return True

    def cancel_all_agent_orders(self, agent_id: int) -> int:
        count = 0
        for order in list(self._orders.values()):
            if order.agent_id == agent_id and order.active:
                order.active = False
                count += 1
        return count

    # ---- Market state queries ---------------------------------------------

    def best_bid(self) -> Optional[float]:
        self._clean_heap(self._bids, is_bid=True)
        return -self._bids[0][0] if self._bids else None

    def best_ask(self) -> Optional[float]:
        self._clean_heap(self._asks, is_bid=False)
        return self._asks[0][0] if self._asks else None

    def mid_price(self) -> float:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        if bb is not None:
            return bb
        if ba is not None:
            return ba
        return self.last_trade_price

    def spread(self) -> float:
        bb = self.best_bid()
        ba = self.best_ask()
        if bb is None or ba is None:
            return float("nan")
        return ba - bb

    def depth(
        self, levels: int = 10
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        bid_map: Dict[float, float] = {}
        ask_map: Dict[float, float] = {}
        for order in self._orders.values():
            if not order.active or order.remaining < EPS:
                continue
            if order.side == SIDE_BID:
                bid_map[order.price] = bid_map.get(order.price, 0.0) + order.remaining
            else:
                ask_map[order.price] = ask_map.get(order.price, 0.0) + order.remaining
        bids = sorted(bid_map.items(), key=lambda x: -x[0])[:levels]
        asks = sorted(ask_map.items(), key=lambda x: x[0])[:levels]
        return bids, asks

    def order_imbalance(self, levels: int = 5) -> float:
        bids, asks = self.depth(levels)
        bid_vol = sum(s for _, s in bids)
        ask_vol = sum(s for _, s in asks)
        total = bid_vol + ask_vol + EPS
        return (bid_vol - ask_vol) / total

    def vwap(self, lookback: int = 50) -> float:
        recent = list(self._recent_trades)[-lookback:]
        if not recent:
            return self.mid_price()
        total_vol = sum(t.size for t in recent)
        if total_vol < EPS:
            return self.mid_price()
        return sum(t.price * t.size for t in recent) / total_vol

    def volatility(self, lookback: int = 100) -> float:
        recent = list(self._recent_trades)[-lookback:]
        if len(recent) < 2:
            return 0.0
        prices = [t.price for t in recent]
        log_returns = [
            math.log(prices[i] / prices[i - 1])
            for i in range(1, len(prices))
            if prices[i - 1] > EPS
        ]
        return float(np.std(log_returns)) if log_returns else 0.0

    def recent_trades(self, n: int = 20) -> List[Trade]:
        return list(self._recent_trades)[-n:]

    def snapshot(self) -> Dict[str, Any]:
        bids, asks = self.depth(5)
        return {
            "asset_id": self.asset_id,
            "best_bid": self.best_bid(),
            "best_ask": self.best_ask(),
            "mid_price": self.mid_price(),
            "spread": self.spread(),
            "last_trade_price": self.last_trade_price,
            "last_trade_size": self.last_trade_size,
            "volume_today": self.volume_today,
            "trade_count": self.trade_count,
            "bid_depth": bids,
            "ask_depth": asks,
            "order_imbalance": self.order_imbalance(),
            "vwap": self.vwap(),
            "volatility": self.volatility(),
        }

    def reset(self, last_price: float = 100.0) -> None:
        self._bids.clear()
        self._asks.clear()
        self._orders.clear()
        self._next_order_id = 0
        self._next_trade_id = 0
        self.last_trade_price = last_price
        self.last_trade_size = 0.0
        self.volume_today = 0.0
        self.trade_count = 0
        self._recent_trades.clear()

    # ---- Internal helpers -------------------------------------------------

    def _round_tick(self, price: float) -> float:
        if self.tick_size <= 0:
            return price
        return round(price / self.tick_size) * self.tick_size

    def _add_to_book(self, order: Order) -> None:
        if order.side == SIDE_BID:
            heapq.heappush(self._bids, (-order.price, order.timestamp, order))
        else:
            heapq.heappush(self._asks, (order.price, order.timestamp, order))

    def _clean_heap(self, heap: List, is_bid: bool) -> None:
        while heap:
            _, _, order = heap[0]
            if not order.active or order.remaining < EPS:
                heapq.heappop(heap)
            else:
                break

    def _match_order(self, aggressor: Order, timestamp: int) -> List[Trade]:
        trades: List[Trade] = []

        if aggressor.side == SIDE_BID:
            passive_heap = self._asks
            is_passive_bid = False
            def price_ok(ap, pp): return ap >= pp
        else:
            passive_heap = self._bids
            is_passive_bid = True
            def price_ok(ap, pp): return ap <= pp

        while aggressor.remaining > EPS and passive_heap:
            self._clean_heap(passive_heap, is_bid=is_passive_bid)
            if not passive_heap:
                break

            if is_passive_bid:
                neg_price, ts, passive = passive_heap[0]
                passive_price = -neg_price
            else:
                passive_price, ts, passive = passive_heap[0]

            if not price_ok(aggressor.price, passive_price):
                break
            if passive.agent_id == aggressor.agent_id:
                heapq.heappop(passive_heap)
                self._add_to_book(passive)
                break

            fill_size = min(aggressor.remaining, passive.remaining)
            fill_price = passive_price

            aggressor.filled += fill_size
            passive.filled += fill_size

            heapq.heappop(passive_heap)
            if passive.remaining >= EPS:
                if is_passive_bid:
                    heapq.heappush(passive_heap, (-passive_price, passive.timestamp, passive))
                else:
                    heapq.heappush(passive_heap, (passive_price, passive.timestamp, passive))
            else:
                passive.active = False

            trade = Trade(
                trade_id=self._next_trade_id,
                asset_id=self.asset_id,
                price=fill_price,
                size=fill_size,
                aggressor_id=aggressor.order_id,
                passive_id=passive.order_id,
                aggressor_agent=aggressor.agent_id,
                passive_agent=passive.agent_id,
                timestamp=timestamp,
                side=aggressor.side,
            )
            self._next_trade_id += 1
            trades.append(trade)
            self._recent_trades.append(trade)
            self.last_trade_price = fill_price
            self.last_trade_size = fill_size
            self.volume_today += fill_size
            self.trade_count += 1

        if aggressor.remaining < EPS:
            aggressor.active = False
        return trades


# ---------------------------------------------------------------------------
# Price processes
# ---------------------------------------------------------------------------

class GBMPriceProcess:
    """Geometric Brownian Motion with optional jumps."""

    def __init__(
        self,
        initial_price: float = 100.0,
        mu: float = 0.0,
        sigma: float = 0.02,
        dt: float = 1.0 / 252.0,
        jump_intensity: float = 0.1,
        jump_mean: float = 0.0,
        jump_std: float = 0.02,
        seed: Optional[int] = None,
    ):
        self.price = initial_price
        self.initial_price = initial_price
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.rng = np.random.default_rng(seed)
        self._path: List[float] = [initial_price]

    def step(self) -> float:
        drift = (self.mu - 0.5 * self.sigma ** 2) * self.dt
        diffusion = self.sigma * math.sqrt(self.dt) * self.rng.standard_normal()
        jump = 0.0
        if self.rng.random() < self.jump_intensity * self.dt:
            jump = self.rng.normal(self.jump_mean, self.jump_std)
        self.price = self.price * math.exp(drift + diffusion + jump)
        self._path.append(self.price)
        return self.price

    def reset(self) -> None:
        self.price = self.initial_price
        self._path = [self.initial_price]


class OrnsteinUhlenbeckProcess:
    """Mean-reverting OU process for spread/inventory signals."""

    def __init__(
        self,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1.0 / 252.0,
        initial: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x = initial
        self.initial = initial
        self.rng = np.random.default_rng(seed)

    def step(self) -> float:
        dx = (
            self.theta * (self.mu - self.x) * self.dt
            + self.sigma * math.sqrt(self.dt) * self.rng.standard_normal()
        )
        self.x += dx
        return self.x

    def reset(self) -> None:
        self.x = self.initial


class CorrelatedAssetProcess:
    """
    Multi-asset correlated GBM.
    Uses Cholesky decomposition for correlation structure.
    """

    def __init__(
        self,
        num_assets: int = DEFAULT_NUM_ASSETS,
        initial_prices: Optional[np.ndarray] = None,
        mu: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None,
        correlation: Optional[np.ndarray] = None,
        dt: float = 1.0 / 252.0,
        jump_intensity: float = 0.05,
        seed: Optional[int] = None,
    ):
        self.num_assets = num_assets
        self.dt = dt
        self.jump_intensity = jump_intensity
        self.rng = np.random.default_rng(seed)

        if initial_prices is not None:
            self.initial_prices = np.asarray(initial_prices, dtype=float)
        else:
            self.initial_prices = np.array([100.0 + 10.0 * i for i in range(num_assets)])
        self.prices = self.initial_prices.copy()

        self.mu = mu if mu is not None else np.zeros(num_assets)
        self.sigma = sigma if sigma is not None else np.full(num_assets, 0.02)

        if correlation is None:
            corr = np.full((num_assets, num_assets), 0.3)
            np.fill_diagonal(corr, 1.0)
            correlation = corr
        self.correlation = correlation
        sig_diag = np.diag(self.sigma)
        self.cov = sig_diag @ correlation @ sig_diag
        self.chol = np.linalg.cholesky(self.cov + EPS * np.eye(num_assets))

        self._paths: List[np.ndarray] = [self.prices.copy()]

    def step(self) -> np.ndarray:
        z = self.rng.standard_normal(self.num_assets)
        corr_z = self.chol @ z
        log_returns = (
            (self.mu - 0.5 * self.sigma ** 2) * self.dt
            + corr_z * math.sqrt(self.dt)
        )
        for i in range(self.num_assets):
            if self.rng.random() < self.jump_intensity * self.dt:
                log_returns[i] += self.rng.normal(0, 0.03)
        self.prices = self.prices * np.exp(log_returns)
        self._paths.append(self.prices.copy())
        return self.prices.copy()

    def reset(self) -> None:
        self.prices = self.initial_prices.copy()
        self._paths = [self.prices.copy()]

    @property
    def path(self) -> np.ndarray:
        return np.array(self._paths)


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

class RewardShaper:
    """
    Shaped reward combining:
    - PnL delta (normalized)
    - Sharpe ratio estimate
    - Drawdown penalty
    - Inventory penalty (quadratic)
    - Transaction cost penalty
    - Market impact penalty
    """

    def __init__(
        self,
        pnl_weight: float = 1.0,
        sharpe_weight: float = 0.1,
        drawdown_weight: float = 0.2,
        inventory_weight: float = 0.05,
        transaction_weight: float = 0.01,
        impact_weight: float = 0.01,
        risk_free_rate: float = 0.02 / 252.0,
        sharpe_window: int = 100,
        normalize_rewards: bool = True,
        reward_clip: float = 10.0,
    ):
        self.pnl_weight = pnl_weight
        self.sharpe_weight = sharpe_weight
        self.drawdown_weight = drawdown_weight
        self.inventory_weight = inventory_weight
        self.transaction_weight = transaction_weight
        self.impact_weight = impact_weight
        self.risk_free_rate = risk_free_rate
        self.sharpe_window = sharpe_window
        self.normalize_rewards = normalize_rewards
        self.reward_clip = reward_clip

        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._alpha = 0.001

    def compute(
        self,
        prev_state: AgentState,
        curr_state: AgentState,
        mid_prices: np.ndarray,
        transaction_costs: float,
        market_impact: float,
    ) -> Tuple[float, Dict[str, float]]:
        prev_equity = prev_state.net_equity(mid_prices)
        curr_equity = curr_state.net_equity(mid_prices)
        pnl_delta = curr_equity - prev_equity
        pnl_reward = self.pnl_weight * pnl_delta / (abs(prev_equity) + EPS)

        sharpe_bonus = 0.0
        if len(curr_state.returns_history) >= 10:
            rets = np.array(curr_state.returns_history[-self.sharpe_window:])
            mean_ret = float(np.mean(rets)) - self.risk_free_rate
            std_ret = float(np.std(rets)) + EPS
            sharpe = mean_ret / std_ret * math.sqrt(252)
            sharpe_bonus = self.sharpe_weight * float(np.tanh(sharpe / 3.0))

        drawdown_penalty = 0.0
        if curr_state.peak_equity > EPS:
            current_dd = (curr_state.peak_equity - curr_equity) / curr_state.peak_equity
            drawdown_penalty = -self.drawdown_weight * max(0.0, current_dd) ** 2

        inv_norm = float(np.sum(curr_state.positions ** 2) ** 0.5)
        inv_penalty = -self.inventory_weight * inv_norm / (DEFAULT_MAX_POSITION + EPS)

        tc_penalty = -self.transaction_weight * transaction_costs / (abs(prev_equity) + EPS)
        impact_penalty = -self.impact_weight * market_impact / (abs(prev_equity) + EPS)

        total = (
            pnl_reward + sharpe_bonus + drawdown_penalty
            + inv_penalty + tc_penalty + impact_penalty
        )

        if self.normalize_rewards:
            total = self._normalize(total)

        total = float(np.clip(total, -self.reward_clip, self.reward_clip))

        components = {
            "pnl": pnl_reward, "sharpe": sharpe_bonus,
            "drawdown": drawdown_penalty, "inventory": inv_penalty,
            "transaction_cost": tc_penalty, "market_impact": impact_penalty,
            "total": total,
        }
        return total, components

    def _normalize(self, reward: float) -> float:
        self._reward_mean += self._alpha * (reward - self._reward_mean)
        self._reward_var += self._alpha * ((reward - self._reward_mean) ** 2 - self._reward_var)
        std = math.sqrt(max(self._reward_var, EPS))
        return (reward - self._reward_mean) / std

    def reset(self) -> None:
        self._reward_mean = 0.0
        self._reward_var = 1.0


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

class ObservationBuilder:
    """
    Constructs per-agent and global observation vectors.

    Per-agent obs: portfolio state + order book features + private signal + time
    Global obs: all per-agent obs concatenated + market summary
    """

    def __init__(
        self,
        num_assets: int,
        num_agents: int,
        ob_levels: int = 5,
        private_signal_noise: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.num_assets = num_assets
        self.num_agents = num_agents
        self.ob_levels = ob_levels
        self.private_signal_noise = private_signal_noise
        self.rng = np.random.default_rng(seed)

        # Dimensions
        self.portfolio_dim = 1 + num_assets + num_assets + 1 + 1  # cash, pos, avg_cost, unreal, real
        self.ob_per_asset = 7 + ob_levels * 4  # 7 scalars + depth levels (price+size)*2 sides
        self.ob_dim = num_assets * self.ob_per_asset
        self.signal_dim = num_assets
        self.time_dim = 4
        self.per_agent_dim = self.portfolio_dim + self.ob_dim + self.signal_dim + self.time_dim

        self.market_summary_dim = num_assets * 5
        self.global_dim = self.per_agent_dim * num_agents + self.market_summary_dim

    def build_per_agent(
        self,
        agent_state: AgentState,
        order_books: List[OrderBook],
        mid_prices: np.ndarray,
        fundamental_values: np.ndarray,
        step: int,
        max_steps: int,
    ) -> np.ndarray:
        parts = []

        # Portfolio
        cash_n = float(np.tanh(agent_state.cash / DEFAULT_INITIAL_CASH))
        pos_n = np.tanh(agent_state.positions / DEFAULT_MAX_POSITION)
        cost_n = np.tanh((agent_state.avg_cost - mid_prices) / (mid_prices + EPS))
        unreal_n = float(np.tanh(agent_state.unrealized_pnl / (DEFAULT_INITIAL_CASH + EPS)))
        real_n = float(np.tanh(agent_state.realized_pnl / (DEFAULT_INITIAL_CASH + EPS)))

        parts.append(np.array([cash_n]))
        parts.append(pos_n.astype(np.float32))
        parts.append(cost_n.astype(np.float32))
        parts.append(np.array([unreal_n]))
        parts.append(np.array([real_n]))

        # Order book features
        for i, ob in enumerate(order_books):
            mid = ob.mid_price()
            bb = ob.best_bid() or mid * 0.999
            ba = ob.best_ask() or mid * 1.001
            sp = ob.spread()
            sp = 0.002 * mid if (sp is None or math.isnan(sp)) else sp
            imb = ob.order_imbalance()
            vw = ob.vwap()
            vol = ob.volatility()

            sp_n = float(np.tanh(sp / (mid + EPS) * 100))
            mid_n = float(np.tanh((mid - mid_prices[i]) / (mid_prices[i] + EPS) * 10))
            vw_n = float(np.tanh((vw - mid) / (mid + EPS) * 10))
            vol_n = float(np.tanh(vol * 100))
            bid_n = float(np.tanh((bb - mid) / (mid + EPS) * 100))
            ask_n = float(np.tanh((ba - mid) / (mid + EPS) * 100))

            ob_feats = [bid_n, ask_n, sp_n, mid_n, float(imb), vw_n, vol_n]

            bids_depth, asks_depth = ob.depth(self.ob_levels)
            for lvl in range(self.ob_levels):
                if lvl < len(bids_depth):
                    p, s = bids_depth[lvl]
                    ob_feats.extend([float(np.tanh((p - mid) / (mid + EPS) * 100)),
                                     float(np.tanh(s / 100))])
                else:
                    ob_feats.extend([0.0, 0.0])
                if lvl < len(asks_depth):
                    p, s = asks_depth[lvl]
                    ob_feats.extend([float(np.tanh((p - mid) / (mid + EPS) * 100)),
                                     float(np.tanh(s / 100))])
                else:
                    ob_feats.extend([0.0, 0.0])

            parts.append(np.array(ob_feats, dtype=np.float32))

        # Private signal
        noise = self.rng.normal(0, self.private_signal_noise, self.num_assets)
        signals = np.tanh(
            (fundamental_values * (1.0 + noise) - mid_prices) / (mid_prices + EPS) * 10
        ).astype(np.float32)
        parts.append(signals)

        # Time features
        t_frac = step / max(max_steps, 1)
        intraday = (step % 390) / 390.0
        parts.append(np.array([
            t_frac,
            math.sin(2 * math.pi * t_frac),
            math.cos(2 * math.pi * t_frac),
            intraday,
        ], dtype=np.float32))

        obs = np.concatenate([p.flatten() for p in parts]).astype(np.float32)
        # Pad or truncate to per_agent_dim
        if len(obs) < self.per_agent_dim:
            obs = np.pad(obs, (0, self.per_agent_dim - len(obs)))
        elif len(obs) > self.per_agent_dim:
            obs = obs[:self.per_agent_dim]
        return obs

    def build_global(
        self,
        all_agent_states: List[AgentState],
        order_books: List[OrderBook],
        mid_prices: np.ndarray,
        fundamental_values: np.ndarray,
        step: int,
        max_steps: int,
        price_history: np.ndarray,
    ) -> np.ndarray:
        per_agent_obs = [
            self.build_per_agent(s, order_books, mid_prices, fundamental_values, step, max_steps)
            for s in all_agent_states
        ]

        market_feats = []
        for i in range(self.num_assets):
            price = mid_prices[i]
            ret = 0.0
            if price_history.shape[0] >= 2:
                prev = price_history[-2, i]
                ret = (price - prev) / (prev + EPS)
            vol = 0.0
            if price_history.shape[0] > 5:
                px = price_history[-20:, i] if price_history.shape[0] >= 20 else price_history[:, i]
                if len(px) > 1:
                    vol = float(np.std(np.diff(px) / (px[:-1] + EPS)))
            volume = order_books[i].volume_today
            trend = 0.0
            if price_history.shape[0] >= 5:
                trend = (price - price_history[-5, i]) / (price_history[-5, i] + EPS)
            market_feats.extend([
                float(np.tanh(price / 100)),
                float(np.tanh(ret * 100)),
                float(np.tanh(vol * 100)),
                float(np.tanh(volume / 1000)),
                float(np.tanh(trend * 50)),
            ])

        global_obs = np.concatenate(
            [o for o in per_agent_obs]
            + [np.array(market_feats, dtype=np.float32)]
        ).astype(np.float32)
        return global_obs


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Market circuit breaker: halts trading on large price moves."""

    def __init__(
        self,
        level1_threshold: float = 0.07,
        level2_threshold: float = 0.13,
        level3_threshold: float = 0.20,
        halt_duration: int = 15,
    ):
        self.level1 = level1_threshold
        self.level2 = level2_threshold
        self.level3 = level3_threshold
        self.halt_duration = halt_duration
        self._halt_until: Dict[int, int] = {}
        self._reference_prices: Dict[int, float] = {}
        self._triggered: Dict[int, int] = {}

    def check(self, asset_id: int, price: float, step: int) -> Tuple[bool, int]:
        if asset_id in self._halt_until and step < self._halt_until[asset_id]:
            return True, self._triggered.get(asset_id, 0)
        ref = self._reference_prices.get(asset_id, price)
        move = abs(price - ref) / (ref + EPS)
        if move >= self.level3:
            self._halt_until[asset_id] = step + self.halt_duration * 4
            self._triggered[asset_id] = 3
            return True, 3
        elif move >= self.level2:
            self._halt_until[asset_id] = step + self.halt_duration * 2
            self._triggered[asset_id] = 2
            return True, 2
        elif move >= self.level1:
            self._halt_until[asset_id] = step + self.halt_duration
            self._triggered[asset_id] = 1
            return True, 1
        return False, 0

    def update_reference(self, asset_id: int, price: float) -> None:
        self._reference_prices[asset_id] = price

    def reset(self, initial_prices: np.ndarray) -> None:
        self._halt_until.clear()
        self._reference_prices = {i: float(p) for i, p in enumerate(initial_prices)}
        self._triggered.clear()


# ---------------------------------------------------------------------------
# Flash crash simulator
# ---------------------------------------------------------------------------

class FlashCrashSimulator:
    """Stochastically injects flash crash events."""

    def __init__(
        self,
        crash_probability: float = 0.0002,
        crash_magnitude: float = 0.05,
        recovery_speed: float = 0.5,
        affected_assets: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        self.crash_probability = crash_probability
        self.crash_magnitude = crash_magnitude
        self.recovery_speed = recovery_speed
        self.affected_assets = affected_assets
        self.rng = np.random.default_rng(seed)
        self._active_crash: Optional[Dict] = None

    def step(self, prices: np.ndarray, step: int) -> Tuple[np.ndarray, bool]:
        if self._active_crash is not None:
            crash = self._active_crash
            recovery_fraction = self.recovery_speed * 0.1
            crash["current"] = crash["current"] + (
                crash["target_recovery"] - crash["current"]
            ) * recovery_fraction
            if np.all(np.abs(crash["current"] - crash["target_recovery"]) < 0.01):
                self._active_crash = None
            return crash["current"].copy(), True

        if self.rng.random() < self.crash_probability:
            assets = self.affected_assets or list(range(len(prices)))
            magnitude = float(self.rng.uniform(0.5, 1.5)) * self.crash_magnitude
            new_prices = prices.copy()
            for i in assets:
                new_prices[i] *= (1.0 - magnitude)
            self._active_crash = {
                "start_step": step,
                "original_prices": prices.copy(),
                "target_recovery": prices * (1.0 - magnitude * 0.3),
                "current": new_prices.copy(),
            }
            logger.info(f"Flash crash at step {step}: magnitude={magnitude:.2%}")
            return new_prices, True
        return prices, False

    def reset(self) -> None:
        self._active_crash = None


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class MultiAssetTradingEnv(gym.Env):
    """
    Full multi-asset, multi-agent trading environment.

    Action space (per agent): Box(-1, 1, shape=(num_assets*4,))
      Each asset: [bid_offset, ask_offset, bid_size_frac, ask_size_frac]

    Observation space: Box(-inf, inf, shape=(per_agent_dim,))

    In MARL mode: step() accepts List[np.ndarray] and returns lists.
    In single-agent mode: normal gymnasium interface.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        num_assets: int = DEFAULT_NUM_ASSETS,
        num_agents: int = DEFAULT_NUM_AGENTS,
        max_steps: int = DEFAULT_MAX_STEPS,
        initial_cash: float = DEFAULT_INITIAL_CASH,
        transaction_cost: float = DEFAULT_TRANSACTION_COST,
        max_position: float = DEFAULT_MAX_POSITION,
        tick_size: float = DEFAULT_TICK_SIZE,
        lot_size: float = DEFAULT_LOT_SIZE,
        reward_shaper_config: Optional[Dict] = None,
        price_process_config: Optional[Dict] = None,
        ob_levels: int = 5,
        enable_circuit_breaker: bool = True,
        enable_flash_crash: bool = True,
        private_signal_noise: float = 0.01,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        marl_mode: bool = True,
        agent_id: int = 0,
    ):
        super().__init__()

        self.num_assets = num_assets
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.render_mode = render_mode
        self.marl_mode = marl_mode
        self.agent_id = agent_id
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

        # Sub-components
        rsc = reward_shaper_config or {}
        self.reward_shaper = RewardShaper(**rsc)

        ppc = price_process_config or {}
        self.price_process = CorrelatedAssetProcess(
            num_assets=num_assets, seed=seed, **ppc
        )

        self.order_books: List[OrderBook] = [
            OrderBook(asset_id=i, tick_size=tick_size)
            for i in range(num_assets)
        ]

        self.obs_builder = ObservationBuilder(
            num_assets=num_assets,
            num_agents=num_agents,
            ob_levels=ob_levels,
            private_signal_noise=private_signal_noise,
            seed=seed,
        )

        self.circuit_breaker: Optional[CircuitBreaker] = (
            CircuitBreaker() if enable_circuit_breaker else None
        )
        self.flash_crash: Optional[FlashCrashSimulator] = (
            FlashCrashSimulator(seed=seed) if enable_flash_crash else None
        )

        # Spaces
        action_dim = num_assets * 4
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        obs_dim = self.obs_builder.per_agent_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # State
        self._step = 0
        self._agent_states: List[AgentState] = []
        self._mid_prices: np.ndarray = np.zeros(num_assets)
        self._fundamental_values: np.ndarray = np.zeros(num_assets)
        self._price_history: List[np.ndarray] = []
        self._halted_assets: set = set()

        self._ou_processes: List[OrnsteinUhlenbeckProcess] = [
            OrnsteinUhlenbeckProcess(seed=seed + i if seed else None)
            for i in range(num_assets)
        ]

        self._episode_rewards: List[List[float]] = [[] for _ in range(num_agents)]

        self.reset()

    # ---- gymnasium interface ----------------------------------------------

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._seed = seed
            self.np_random = np.random.default_rng(seed)

        self._step = 0
        self.price_process.reset()
        self._mid_prices = self.price_process.prices.copy()
        self._fundamental_values = self._mid_prices.copy()
        self._price_history = [self._mid_prices.copy()]

        for ob in self.order_books:
            ob.reset(float(self._mid_prices[ob.asset_id]))

        if self.circuit_breaker:
            self.circuit_breaker.reset(self._mid_prices)
        if self.flash_crash:
            self.flash_crash.reset()

        self._agent_states = [
            AgentState(
                agent_id=i,
                cash=self.initial_cash,
                positions=np.zeros(self.num_assets),
                avg_cost=self._mid_prices.copy(),
                peak_equity=self.initial_cash,
            )
            for i in range(self.num_agents)
        ]

        for p in self._ou_processes:
            p.reset()

        self.reward_shaper.reset()
        self._episode_rewards = [[] for _ in range(self.num_agents)]
        self._halted_assets = set()

        obs = self._get_obs(0)
        info = self._get_info(0)
        return obs, info

    def step(self, action: Union[np.ndarray, List[np.ndarray]]) -> Tuple:
        if self.marl_mode:
            return self._marl_step(action)
        else:
            actions = [np.zeros(self.num_assets * 4, dtype=np.float32)] * self.num_agents
            actions[self.agent_id] = np.asarray(action, dtype=np.float32)
            obs_list, rew_list, term_list, trunc_list, info_list = self._marl_step(actions)
            return (
                obs_list[self.agent_id],
                rew_list[self.agent_id],
                term_list[self.agent_id],
                trunc_list[self.agent_id],
                info_list[self.agent_id],
            )

    def _marl_step(
        self, actions: List[np.ndarray]
    ) -> Tuple[List, List, List, List, List]:
        self._step += 1

        prev_states = [deepcopy(s) for s in self._agent_states]

        # Advance price
        new_prices = self.price_process.step()
        if self.flash_crash:
            new_prices, _ = self.flash_crash.step(new_prices, self._step)
        self._mid_prices = new_prices
        self._price_history.append(new_prices.copy())

        # Cancel existing orders
        for agent_id in range(self.num_agents):
            for ob in self.order_books:
                ob.cancel_all_agent_orders(agent_id)

        # Noise traders
        self._submit_noise_orders()

        # Process agent actions (randomized order)
        agent_tc = [0.0] * self.num_agents
        agent_impact = [0.0] * self.num_agents
        order = list(range(self.num_agents))
        self.np_random.shuffle(order)

        for aid in order:
            if aid >= len(actions):
                continue
            action = np.asarray(actions[aid], dtype=np.float32)
            tc, imp = self._process_action(aid, action)
            agent_tc[aid] = tc
            agent_impact[aid] = imp

        # Update portfolio states
        for aid, state in enumerate(self._agent_states):
            state.update_unrealized(self._mid_prices)
            equity = state.net_equity(self._mid_prices)
            if equity > state.peak_equity:
                state.peak_equity = equity
            if state.equity_history:
                prev_eq = state.equity_history[-1]
                ret = (equity - prev_eq) / (prev_eq + EPS)
                state.returns_history.append(float(ret))
            state.equity_history.append(float(equity))

        # Circuit breaker
        self._halted_assets.clear()
        if self.circuit_breaker:
            for i in range(self.num_assets):
                halted, _ = self.circuit_breaker.check(i, float(self._mid_prices[i]), self._step)
                if halted:
                    self._halted_assets.add(i)

        # Rewards
        rewards = []
        for aid in range(self.num_agents):
            r, _ = self.reward_shaper.compute(
                prev_states[aid], self._agent_states[aid],
                self._mid_prices, agent_tc[aid], agent_impact[aid],
            )
            rewards.append(r)
            self._episode_rewards[aid].append(r)

        # Termination
        terminated = [self._check_terminated(i) for i in range(self.num_agents)]
        truncated = [self._step >= self.max_steps] * self.num_agents

        observations = [self._get_obs(i) for i in range(self.num_agents)]
        infos = [self._get_info(i) for i in range(self.num_agents)]

        return observations, rewards, terminated, truncated, infos

    def _process_action(self, agent_id: int, action: np.ndarray) -> Tuple[float, float]:
        state = self._agent_states[agent_id]
        total_tc = 0.0
        total_impact = 0.0

        for i in range(self.num_assets):
            if i in self._halted_assets:
                continue
            mid = float(self._mid_prices[i])
            if mid <= 0:
                continue

            bid_offset = float(np.clip(action[i * 4 + 0], -1, 1))
            ask_offset = float(np.clip(action[i * 4 + 1], -1, 1))
            bid_size_frac = float(np.clip((action[i * 4 + 2] + 1) / 2, 0, 1))
            ask_size_frac = float(np.clip((action[i * 4 + 3] + 1) / 2, 0, 1))

            max_offset = mid * 0.02
            bid_price = max(mid + bid_offset * max_offset, self.tick_size)
            ask_price = max(mid + ask_offset * max_offset, self.tick_size)
            if ask_price <= bid_price:
                ask_price = bid_price + self.tick_size

            max_bid_size = min(
                bid_size_frac * self.max_position,
                state.cash / (bid_price + EPS),
            )
            max_ask_size = min(
                ask_size_frac * self.max_position,
                float(state.positions[i]) + self.max_position,
            )
            bid_size = max(0.0, max_bid_size)
            ask_size = max(0.0, max_ask_size)

            ob = self.order_books[i]

            if bid_size >= self.lot_size:
                _, trades = ob.submit_limit_order(
                    agent_id, SIDE_BID, bid_price, bid_size, self._step
                )
                for trade in trades:
                    tc, imp = self._settle_trade(trade, agent_id, is_buyer=True)
                    total_tc += tc
                    total_impact += imp

            if ask_size >= self.lot_size:
                _, trades = ob.submit_limit_order(
                    agent_id, SIDE_ASK, ask_price, ask_size, self._step
                )
                for trade in trades:
                    tc, imp = self._settle_trade(trade, agent_id, is_buyer=False)
                    total_tc += tc
                    total_impact += imp

        return total_tc, total_impact

    def _settle_trade(
        self, trade: Trade, agent_id: int, is_buyer: bool
    ) -> Tuple[float, float]:
        state = self._agent_states[agent_id]
        i = trade.asset_id
        size = trade.size
        price = trade.price
        tc = price * size * self.transaction_cost

        if is_buyer:
            cost = price * size + tc
            if state.cash < cost:
                return tc, 0.0
            old_pos = float(state.positions[i])
            new_pos = old_pos + size
            if new_pos > EPS:
                state.avg_cost[i] = (old_pos * state.avg_cost[i] + size * price) / new_pos
            state.positions[i] = min(new_pos, self.max_position)
            state.cash -= cost
        else:
            old_pos = float(state.positions[i])
            realized = (price - state.avg_cost[i]) * min(size, max(old_pos, 0.0))
            state.realized_pnl += realized
            state.positions[i] = old_pos - size
            state.cash += price * size - tc

        state.total_trades += 1
        state.total_volume += size
        impact = price * size * math.sqrt(size / (self.max_position + EPS)) * 0.001
        return tc, impact

    def _submit_noise_orders(self) -> None:
        for i, ob in enumerate(self.order_books):
            mid = float(self._mid_prices[i])
            signal = self._ou_processes[i].step()
            for _ in range(3):
                side = SIDE_BID if self.np_random.random() > 0.5 else SIDE_ASK
                offset = abs(float(self.np_random.normal(0, 0.002))) * mid
                price = mid - offset if side == SIDE_BID else mid + offset
                size = abs(float(self.np_random.normal(10, 5))) + 1.0
                ob.submit_limit_order(
                    agent_id=-1, side=side,
                    price=max(price, self.tick_size), size=size,
                    timestamp=self._step,
                )
            if self.np_random.random() < 0.1:
                side = SIDE_BID if signal > 0 else SIDE_ASK
                size = abs(float(self.np_random.normal(5, 2))) + 1.0
                ob.submit_market_order(-1, side, size, self._step)

    def _check_terminated(self, agent_id: int) -> bool:
        state = self._agent_states[agent_id]
        equity = state.net_equity(self._mid_prices)
        return equity < self.initial_cash * 0.1

    def _get_obs(self, agent_id: int) -> np.ndarray:
        return self.obs_builder.build_per_agent(
            agent_state=self._agent_states[agent_id],
            order_books=self.order_books,
            mid_prices=self._mid_prices,
            fundamental_values=self._fundamental_values,
            step=self._step,
            max_steps=self.max_steps,
        )

    def get_global_obs(self) -> np.ndarray:
        return self.obs_builder.build_global(
            all_agent_states=self._agent_states,
            order_books=self.order_books,
            mid_prices=self._mid_prices,
            fundamental_values=self._fundamental_values,
            step=self._step,
            max_steps=self.max_steps,
            price_history=np.array(self._price_history),
        )

    def get_state(self) -> np.ndarray:
        return self.get_global_obs()

    def _get_info(self, agent_id: int = 0) -> Dict:
        state = self._agent_states[agent_id]
        equity = state.net_equity(self._mid_prices)
        return {
            "step": self._step,
            "agent_id": agent_id,
            "cash": state.cash,
            "positions": state.positions.tolist(),
            "equity": equity,
            "realized_pnl": state.realized_pnl,
            "unrealized_pnl": state.unrealized_pnl,
            "total_trades": state.total_trades,
            "mid_prices": self._mid_prices.tolist(),
            "episode_reward": sum(self._episode_rewards[agent_id]),
        }

    def get_all_observations(self) -> List[np.ndarray]:
        return [self._get_obs(i) for i in range(self.num_agents)]

    def get_all_infos(self) -> List[Dict]:
        return [self._get_info(i) for i in range(self.num_agents)]

    @property
    def state_dim(self) -> int:
        return self.obs_builder.global_dim

    @property
    def obs_dim(self) -> int:
        return self.obs_builder.per_agent_dim

    @property
    def action_dim(self) -> int:
        return self.num_assets * 4

    def render(self) -> Optional[Any]:
        if self.render_mode in ("human", "ansi"):
            lines = [
                f"Step: {self._step}/{self.max_steps}",
                f"Assets: {self.num_assets}, Agents: {self.num_agents}",
            ]
            for i in range(self.num_assets):
                ob = self.order_books[i]
                lines.append(
                    f"  Asset {i}: mid={ob.mid_price():.4f} "
                    f"spread={ob.spread():.6f} vol={ob.volume_today:.1f}"
                )
            for i in range(min(4, self.num_agents)):
                s = self._agent_states[i]
                eq = s.net_equity(self._mid_prices)
                lines.append(
                    f"  Agent {i}: equity={eq:.2f} pnl={s.realized_pnl:.2f} trades={s.total_trades}"
                )
            text = "\n".join(lines)
            if self.render_mode == "human":
                print(text)
            return text
        return None

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class SingleAgentWrapper(gym.Wrapper):
    """Gymnasium-compatible single-agent wrapper around multi-agent env."""

    def __init__(self, env: MultiAssetTradingEnv, agent_id: int = 0):
        super().__init__(env)
        self.agent_id = agent_id
        env.marl_mode = False
        env.agent_id = agent_id

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


# ---------------------------------------------------------------------------
# Vectorized environment
# ---------------------------------------------------------------------------

class VecTradingEnv:
    """Synchronous vectorized environment."""

    def __init__(self, num_envs: int, env_config: Optional[Dict] = None):
        self.num_envs = num_envs
        env_config = env_config or {}
        self.envs = [
            MultiAssetTradingEnv(seed=i, **env_config)
            for i in range(num_envs)
        ]
        self.num_agents = self.envs[0].num_agents

    def reset(self) -> List[List[np.ndarray]]:
        results = []
        for env in self.envs:
            env.reset()
            results.append(env.get_all_observations())
        return results

    def step(
        self, actions: List[List[np.ndarray]]
    ) -> Tuple[List, List, List, List, List]:
        all_obs, all_rew, all_term, all_trunc, all_info = [], [], [], [], []
        for env, env_acts in zip(self.envs, actions):
            obs, rew, term, trunc, info = env._marl_step(env_acts)
            all_obs.append(obs)
            all_rew.append(rew)
            all_term.append(term)
            all_trunc.append(trunc)
            all_info.append(info)
        return all_obs, all_rew, all_term, all_trunc, all_info

    def get_global_states(self) -> List[np.ndarray]:
        return [env.get_state() for env in self.envs]

    def close(self) -> None:
        for env in self.envs:
            env.close()


# ---------------------------------------------------------------------------
# Microstructure analyzer
# ---------------------------------------------------------------------------

class MarketMicrostructureAnalyzer:
    """Computes market microstructure metrics."""

    def __init__(self, env: MultiAssetTradingEnv):
        self.env = env

    def price_impact(self, asset_id: int, lookback: int = 50) -> float:
        ob = self.env.order_books[asset_id]
        trades = ob.recent_trades(lookback)
        if len(trades) < 5:
            return 0.0
        prices = np.array([t.price for t in trades])
        sizes = np.array([t.size * (1 if t.side == SIDE_BID else -1) for t in trades])
        price_changes = np.diff(prices)
        if len(price_changes) == 0:
            return 0.0
        order_flow = sizes[:-1]
        var_of = float(np.var(order_flow))
        if var_of < EPS:
            return 0.0
        cov = np.cov(order_flow, price_changes)
        return float(cov[0, 1] / (var_of + EPS))

    def effective_spread(self, asset_id: int, lookback: int = 50) -> float:
        ob = self.env.order_books[asset_id]
        trades = ob.recent_trades(lookback)
        if not trades:
            return float("nan")
        mid = ob.mid_price()
        if mid <= 0:
            return float("nan")
        spreads = [2 * abs(t.price - mid) / mid for t in trades]
        return float(np.mean(spreads))

    def summary(self) -> Dict[str, Any]:
        result = {}
        for i in range(self.env.num_assets):
            ob = self.env.order_books[i]
            result[f"asset_{i}"] = {
                "price_impact": self.price_impact(i),
                "effective_spread": self.effective_spread(i),
                "mid_price": ob.mid_price(),
                "spread": ob.spread(),
                "volume": ob.volume_today,
                "volatility": ob.volatility(),
                "order_imbalance": ob.order_imbalance(),
            }
        return result


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_env(config: Optional[Dict] = None, marl: bool = True) -> MultiAssetTradingEnv:
    config = config or {}
    return MultiAssetTradingEnv(marl_mode=marl, **config)


def make_single_agent_env(agent_id: int = 0, config: Optional[Dict] = None) -> SingleAgentWrapper:
    env = make_env(config, marl=False)
    return SingleAgentWrapper(env, agent_id=agent_id)


def validate_env(env: MultiAssetTradingEnv, n_steps: int = 20) -> bool:
    env.reset()
    for _ in range(n_steps):
        actions = [env.action_space.sample() for _ in range(env.num_agents)]
        obs, rew, term, trunc, info = env._marl_step(actions)
        assert len(obs) == env.num_agents
        for o in obs:
            assert not np.any(np.isnan(o)), "NaN in observation"
        for r in rew:
            assert not math.isnan(r), "NaN in reward"
    return True


__all__ = [
    "Order", "Trade", "AgentState", "OrderBook",
    "GBMPriceProcess", "OrnsteinUhlenbeckProcess", "CorrelatedAssetProcess",
    "RewardShaper", "ObservationBuilder", "CircuitBreaker", "FlashCrashSimulator",
    "MultiAssetTradingEnv", "SingleAgentWrapper", "VecTradingEnv",
    "MarketMicrostructureAnalyzer", "make_env", "make_single_agent_env", "validate_env",
    "BUY", "SELL", "HOLD", "SIDE_BID", "SIDE_ASK",
    "DEFAULT_NUM_ASSETS", "DEFAULT_NUM_AGENTS", "DEFAULT_MAX_STEPS", "DEFAULT_INITIAL_CASH",
]
