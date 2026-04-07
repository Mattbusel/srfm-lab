"""
research/simulation/agent_based_model.py

Agent-based market simulator for SRFM research. Implements heterogeneous
agent populations with momentum, mean-reversion, market-making, noise, and
informed-trader archetypes. Supports calibration of agent proportions to
match empirical return distribution properties.

Usage:
    market = AgentBasedMarket(initial_price=100.0, price_impact=0.0001)
    market.add_agent(MomentumAgent(threshold=0.02), n_instances=5)
    market.add_agent(MarketMakerAgent(spread=0.002), n_instances=2)
    result = market.run(n_steps=252)
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats, optimize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """A single order submitted by an agent."""
    agent_id: str
    side: str          # "buy" or "sell"
    qty: float         # positive quantity
    order_type: str = "market"   # "market" or "limit"
    limit_price: Optional[float] = None

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise ValueError(f"Invalid side: {self.side}")
        if self.qty <= 0:
            raise ValueError(f"Order qty must be positive, got {self.qty}")
        if self.order_type not in ("market", "limit"):
            raise ValueError(f"Invalid order_type: {self.order_type}")


@dataclass
class Fill:
    """Execution report returned to an agent after an order fills."""
    agent_id: str
    side: str
    qty: float
    fill_price: float
    step: int


@dataclass
class MarketState:
    """Observable market state passed to each agent at every step."""
    price: float          # current mid price
    volume: float         # volume traded this step
    spread: float         # bid-ask spread (absolute)
    informed_fraction: float  # fraction of orders from informed traders (0-1)
    day: int              # simulation day index
    hour: float           # fractional hour within trading day (0-6.5)
    price_history: List[float] = field(default_factory=list)
    returns_history: List[float] = field(default_factory=list)

    def last_n_returns(self, n: int) -> NDArray[np.float64]:
        """Return the last n log-returns as a numpy array."""
        hist = self.returns_history
        if len(hist) == 0:
            return np.zeros(n)
        arr = np.array(hist[-n:], dtype=np.float64)
        if len(arr) < n:
            arr = np.concatenate([np.zeros(n - len(arr)), arr])
        return arr


@dataclass
class SimulationResult:
    """Output of AgentBasedMarket.run()."""
    price_series: NDArray[np.float64]
    volume_series: NDArray[np.float64]
    spread_series: NDArray[np.float64]
    agent_pnls: Dict[str, float]
    returns: NDArray[np.float64]
    n_steps: int


@dataclass
class AgentPopulationConfig:
    """Agent population proportions from calibration."""
    noise_fraction: float
    momentum_fraction: float
    mean_reversion_fraction: float
    informed_fraction: float
    market_maker_fraction: float
    momentum_threshold: float
    mean_reversion_z: float
    noise_sigma: float

    def __post_init__(self) -> None:
        total = (
            self.noise_fraction
            + self.momentum_fraction
            + self.mean_reversion_fraction
            + self.informed_fraction
            + self.market_maker_fraction
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Agent fractions must sum to 1.0, got {total:.6f}"
            )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class MarketAgent(ABC):
    """Abstract base class for all market agents."""

    def __init__(self, agent_id: str, initial_cash: float = 100_000.0) -> None:
        self.agent_id = agent_id
        self.cash = initial_cash
        self.position = 0.0        # shares held (can be negative = short)
        self.realized_pnl = 0.0
        self._fills: List[Fill] = []

    @abstractmethod
    def decide(self, market_state: MarketState) -> Optional[Order]:
        """Given current market state, return an Order or None."""
        ...

    def on_fill(self, fill: Fill) -> None:
        """Called by the market when an order is filled."""
        self._fills.append(fill)
        sign = 1.0 if fill.side == "buy" else -1.0
        cost = sign * fill.qty * fill.fill_price
        self.cash -= cost
        self.position += sign * fill.qty

    def mark_to_market(self, price: float) -> float:
        """Compute total equity: cash + mark-to-market of position."""
        return self.cash + self.position * price

    def pnl(self, current_price: float) -> float:
        """Total unrealized + realized PnL relative to initial cash."""
        return self.mark_to_market(current_price) - 100_000.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id})"


# ---------------------------------------------------------------------------
# Concrete agent types
# ---------------------------------------------------------------------------

class MomentumAgent(MarketAgent):
    """
    Trend-following agent. Buys when 5-day cumulative return exceeds
    `threshold` and sells when it falls below `-threshold`. Uses a fixed
    fractional-of-cash sizing rule.
    """

    def __init__(
        self,
        agent_id: str,
        threshold: float = 0.02,
        lookback: int = 5,
        size_fraction: float = 0.10,
        initial_cash: float = 100_000.0,
    ) -> None:
        super().__init__(agent_id, initial_cash)
        self.threshold = threshold
        self.lookback = lookback
        self.size_fraction = size_fraction
        self._cooldown = 0    # steps remaining before next order allowed

    def decide(self, market_state: MarketState) -> Optional[Order]:
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        returns = market_state.last_n_returns(self.lookback)
        cum_return = float(np.sum(returns))

        price = market_state.price
        if price <= 0:
            return None

        max_shares = (self.cash * self.size_fraction) / price

        if cum_return > self.threshold and max_shares > 0.01:
            self._cooldown = 2
            return Order(
                agent_id=self.agent_id,
                side="buy",
                qty=round(max_shares, 4),
                order_type="market",
            )
        elif cum_return < -self.threshold and self.position > 0.01:
            sell_qty = min(self.position, round(max_shares, 4))
            if sell_qty > 0.01:
                self._cooldown = 2
                return Order(
                    agent_id=self.agent_id,
                    side="sell",
                    qty=sell_qty,
                    order_type="market",
                )
        return None


class MeanReversionAgent(MarketAgent):
    """
    Statistical arbitrage agent. Buys when z-score of price drops below
    `-z_threshold` (oversold) and sells when it rises above `+z_threshold`
    (overbought). Z-score computed over a rolling window.
    """

    def __init__(
        self,
        agent_id: str,
        z_threshold: float = 2.0,
        lookback: int = 20,
        size_fraction: float = 0.10,
        initial_cash: float = 100_000.0,
    ) -> None:
        super().__init__(agent_id, initial_cash)
        self.z_threshold = z_threshold
        self.lookback = lookback
        self.size_fraction = size_fraction

    def _compute_zscore(self, market_state: MarketState) -> Optional[float]:
        hist = market_state.price_history
        if len(hist) < self.lookback:
            return None
        window = np.array(hist[-self.lookback:], dtype=np.float64)
        mu = float(np.mean(window))
        sigma = float(np.std(window, ddof=1))
        if sigma < 1e-12:
            return None
        return (market_state.price - mu) / sigma

    def decide(self, market_state: MarketState) -> Optional[Order]:
        z = self._compute_zscore(market_state)
        if z is None:
            return None

        price = market_state.price
        if price <= 0:
            return None

        max_shares = (self.cash * self.size_fraction) / price

        if z < -self.z_threshold and max_shares > 0.01:
            return Order(
                agent_id=self.agent_id,
                side="buy",
                qty=round(max_shares, 4),
                order_type="market",
            )
        elif z > self.z_threshold and self.position > 0.01:
            sell_qty = min(self.position, round(max_shares, 4))
            if sell_qty > 0.01:
                return Order(
                    agent_id=self.agent_id,
                    side="sell",
                    qty=sell_qty,
                    order_type="market",
                )
        return None


class MarketMakerAgent(MarketAgent):
    """
    Passive liquidity provider. Posts simultaneous bid and ask orders
    symmetrically around the current mid-price. Earns the half-spread on
    each round-trip. Quotes are simplified to signed market orders that
    execute at mid +/- spread/2 (approximated via a single net signed order
    each step based on inventory skew).
    """

    def __init__(
        self,
        agent_id: str,
        spread: float = 0.002,       # as fraction of mid price
        quote_size: float = 100.0,   # shares per side
        max_inventory: float = 500.0,
        initial_cash: float = 100_000.0,
    ) -> None:
        super().__init__(agent_id, initial_cash)
        self.spread = spread
        self.quote_size = quote_size
        self.max_inventory = max_inventory
        self._step_side = True   # alternates buy/sell to simulate two-sided quoting

    def decide(self, market_state: MarketState) -> Optional[Order]:
        """
        Simplified two-sided quoting: alternate buy/sell each step,
        skewing toward inventory reduction when position is large.
        """
        price = market_state.price
        if price <= 0:
            return None

        # inventory skew: prefer to reduce position when near limit
        inv_ratio = self.position / self.max_inventory if self.max_inventory > 0 else 0.0
        inv_ratio = max(-1.0, min(1.0, inv_ratio))

        # determine side based on alternation and inventory skew
        if abs(self.position) >= self.max_inventory * 0.9:
            # forced reduction
            side = "sell" if self.position > 0 else "buy"
        else:
            side = "buy" if self._step_side else "sell"

        self._step_side = not self._step_side

        # size: quote_size scaled down when inventory is large
        qty = self.quote_size * (1.0 - 0.5 * abs(inv_ratio))
        qty = max(1.0, round(qty, 4))

        # check cash constraint for buys
        if side == "buy" and self.cash < qty * price * 1.05:
            if self.cash < price:
                return None
            qty = round(self.cash * 0.95 / price, 4)

        return Order(
            agent_id=self.agent_id,
            side=side,
            qty=qty,
            order_type="market",
        )

    def on_fill(self, fill: Fill) -> None:
        """Override to also book the spread income as realized PnL."""
        super().on_fill(fill)
        # approximate spread earned per fill
        spread_income = fill.qty * fill.fill_price * self.spread * 0.5
        self.realized_pnl += spread_income
        self.cash += spread_income


class NoiseAgent(MarketAgent):
    """
    Uninformed random trader. Submits buy/sell orders with sizes drawn from
    a half-normal distribution. Acts as the liquidity demand baseline.
    """

    def __init__(
        self,
        agent_id: str,
        mean_size: float = 50.0,
        sigma_size: float = 30.0,
        activity_prob: float = 0.3,
        initial_cash: float = 100_000.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(agent_id, initial_cash)
        self.mean_size = mean_size
        self.sigma_size = sigma_size
        self.activity_prob = activity_prob
        self._rng = rng if rng is not None else np.random.default_rng()

    def decide(self, market_state: MarketState) -> Optional[Order]:
        if self._rng.random() > self.activity_prob:
            return None

        size = abs(self._rng.normal(self.mean_size, self.sigma_size))
        size = max(1.0, round(size, 4))
        side = "buy" if self._rng.random() < 0.5 else "sell"

        price = market_state.price
        if side == "buy" and self.cash < size * price * 0.1:
            return None

        return Order(
            agent_id=self.agent_id,
            side=side,
            qty=size,
            order_type="market",
        )


class InformedTrader(MarketAgent):
    """
    Agent with privileged knowledge of the true fundamental value. Trades
    aggressively when the market price diverges from true value, driving
    price discovery. True value follows a random walk updated externally.
    """

    def __init__(
        self,
        agent_id: str,
        true_value: float,
        aggressiveness: float = 0.5,
        size_fraction: float = 0.15,
        initial_cash: float = 100_000.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(agent_id, initial_cash)
        self.true_value = true_value
        self.aggressiveness = aggressiveness    # fraction of mispricing to trade on
        self.size_fraction = size_fraction
        self._rng = rng if rng is not None else np.random.default_rng()

    def update_true_value(self, new_value: float) -> None:
        """Called by the market to update the fundamental value each step."""
        self.true_value = new_value

    def decide(self, market_state: MarketState) -> Optional[Order]:
        price = market_state.price
        if price <= 0:
            return None

        mispricing = (self.true_value - price) / price

        # minimum signal to trade: 0.1% mispricing
        if abs(mispricing) < 0.001:
            return None

        side = "buy" if mispricing > 0 else "sell"

        # size proportional to mispricing and aggressiveness
        fraction = min(
            self.size_fraction,
            self.size_fraction * abs(mispricing) / 0.01 * self.aggressiveness,
        )

        if side == "buy":
            if self.cash < price:
                return None
            qty = round((self.cash * fraction) / price, 4)
        else:
            qty = round(self.position * fraction, 4)
            if qty < 0.01:
                return None

        if qty <= 0:
            return None

        return Order(
            agent_id=self.agent_id,
            side=side,
            qty=qty,
            order_type="market",
        )


# ---------------------------------------------------------------------------
# Agent-based market
# ---------------------------------------------------------------------------

class AgentBasedMarket:
    """
    Simulates a stylized financial market with heterogeneous agents.

    At each time step:
      1. Each agent observes the current MarketState.
      2. Each agent optionally submits an Order.
      3. Orders are cleared in random sequence (uniform priority).
      4. Price is updated based on signed order flow.
      5. Spread adjusts based on informed-trader participation.
    """

    def __init__(
        self,
        initial_price: float = 100.0,
        price_impact: float = 1e-4,
        base_spread: float = 0.002,
        true_value_vol: float = 0.001,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.price = initial_price
        self.price_impact = price_impact
        self.base_spread = base_spread
        self.true_value_vol = true_value_vol
        self._rng = rng if rng is not None else np.random.default_rng()

        self._agents: List[MarketAgent] = []
        self._price_history: List[float] = [initial_price]
        self._returns_history: List[float] = []
        self._true_value = initial_price

    def add_agent(self, agent: MarketAgent, n_instances: int = 1) -> None:
        """
        Add one or more copies of an agent type to the market.
        Each copy gets a unique agent_id based on the original id.
        """
        for i in range(n_instances):
            if i == 0:
                self._agents.append(agent)
            else:
                # create a shallow copy with a new id
                import copy
                clone = copy.deepcopy(agent)
                clone.agent_id = f"{agent.agent_id}_{i}"
                self._agents.append(clone)

    def _build_state(self, step: int, spread: float, informed_frac: float) -> MarketState:
        bars_per_day = 26    # 15-min bars in a 6.5-hour session
        day = step // bars_per_day
        bar_in_day = step % bars_per_day
        hour = bar_in_day * 0.25    # 15 minutes = 0.25 hours

        last_vol = self._price_history[-1] if len(self._price_history) > 1 else 0.0

        return MarketState(
            price=self.price,
            volume=last_vol,
            spread=spread,
            informed_fraction=informed_frac,
            day=day,
            hour=hour,
            price_history=list(self._price_history),
            returns_history=list(self._returns_history),
        )

    def _compute_spread(self, informed_frac: float) -> float:
        """Spread widens with informed-trader participation (adverse selection)."""
        return self.base_spread * (1.0 + 3.0 * informed_frac)

    def _update_true_value(self) -> None:
        """Random walk for fundamental value."""
        shock = self._rng.normal(0.0, self.true_value_vol)
        self._true_value *= math.exp(shock)
        # propagate to informed traders
        for agent in self._agents:
            if isinstance(agent, InformedTrader):
                agent.update_true_value(self._true_value)

    def run(self, n_steps: int, dt: float = 1.0 / 252) -> SimulationResult:
        """
        Run the market simulation for n_steps time steps.

        Parameters
        --------
        n_steps : int
            Number of simulation steps.
        dt : float
            Time increment per step in years (default 1/252 = 1 trading day).

        Returns
        -----
        SimulationResult
        """
        price_series = np.zeros(n_steps + 1, dtype=np.float64)
        volume_series = np.zeros(n_steps, dtype=np.float64)
        spread_series = np.zeros(n_steps, dtype=np.float64)
        price_series[0] = self.price

        n_agents = len(self._agents)
        if n_agents == 0:
            logger.warning("No agents registered; market will not move.")

        for step in range(n_steps):
            self._update_true_value()

            # count informed traders to compute adverse selection spread
            n_informed = sum(
                1 for a in self._agents if isinstance(a, InformedTrader)
            )
            informed_frac = n_informed / max(n_agents, 1)
            spread = self._compute_spread(informed_frac)

            state = self._build_state(step, spread, informed_frac)

            # collect orders from all agents
            orders: List[Order] = []
            for agent in self._agents:
                try:
                    order = agent.decide(state)
                    if order is not None:
                        orders.append(order)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Agent %s raised: %s", agent.agent_id, exc)

            # shuffle order sequence (random priority)
            idx = self._rng.permutation(len(orders))
            orders = [orders[i] for i in idx]

            # compute signed flow and total volume
            signed_flow = 0.0
            total_volume = 0.0
            for order in orders:
                sign = 1.0 if order.side == "buy" else -1.0
                signed_flow += sign * order.qty
                total_volume += order.qty

            # price impact
            old_price = self.price
            self.price = max(
                1e-6,
                self.price * math.exp(self.price_impact * signed_flow),
            )

            # fill all orders at new price (simplified: single clearing price)
            for order in orders:
                fill = Fill(
                    agent_id=order.agent_id,
                    side=order.side,
                    qty=order.qty,
                    fill_price=self.price,
                    step=step,
                )
                # dispatch fill to owning agent
                for agent in self._agents:
                    if agent.agent_id == order.agent_id:
                        agent.on_fill(fill)
                        break

            # record
            price_series[step + 1] = self.price
            volume_series[step] = total_volume
            spread_series[step] = spread

            log_ret = math.log(self.price / old_price) if old_price > 0 else 0.0
            self._price_history.append(self.price)
            self._returns_history.append(log_ret)

        returns = np.diff(np.log(price_series + 1e-12))
        agent_pnls = {
            a.agent_id: a.pnl(self.price) for a in self._agents
        }

        return SimulationResult(
            price_series=price_series,
            volume_series=volume_series,
            spread_series=spread_series,
            agent_pnls=agent_pnls,
            returns=returns,
            n_steps=n_steps,
        )


# ---------------------------------------------------------------------------
# Calibration: fit agent population to empirical return distribution
# ---------------------------------------------------------------------------

def _return_moments(returns: NDArray[np.float64]) -> Dict[str, float]:
    """Compute distributional moments and stylized facts from a return series."""
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns))  # excess kurtosis

    # autocorrelation of squared returns (ARCH effect proxy)
    r2 = returns ** 2
    ac1 = float(np.corrcoef(r2[:-1], r2[1:])[0, 1]) if len(r2) > 2 else 0.0

    # tail thickness: ratio of 99th to 95th percentile of |returns|
    abs_r = np.abs(returns)
    tail = float(np.percentile(abs_r, 99) / (np.percentile(abs_r, 95) + 1e-12))

    return {
        "mean": mu,
        "std": sigma,
        "skew": skew,
        "kurt": kurt,
        "arch_effect": ac1,
        "tail_ratio": tail,
    }


def fit_to_empirical(
    empirical_returns: NDArray[np.float64],
    total_agents: int = 100,
) -> AgentPopulationConfig:
    """
    Calibrate agent population fractions to match empirical return properties.

    Strategy:
      - High kurtosis / fat tails       -> more informed traders
      - Strong ARCH effect              -> more momentum traders
      - Near-zero autocorrelation       -> more noise / mean-reversion traders
      - Negative skew                   -> asymmetric momentum weighting

    Parameters
    --------
    empirical_returns : array of log-returns
    total_agents : reference population size (not used directly, for scaling)

    Returns
    -----
    AgentPopulationConfig
    """
    moments = _return_moments(empirical_returns)

    kurt = moments["kurt"]
    arch = moments["arch_effect"]
    tail = moments["tail_ratio"]
    sigma = moments["std"]

    # informed fraction: driven by excess kurtosis (heavier tails = more info asymmetry)
    informed_frac = float(np.clip(0.05 + 0.05 * (kurt / 3.0), 0.02, 0.25))

    # momentum fraction: driven by ARCH effect (volatility clustering)
    momentum_frac = float(np.clip(0.10 + 0.20 * max(arch, 0.0), 0.05, 0.35))

    # mean-reversion fraction: inversely related to momentum
    mr_frac = float(np.clip(0.15 - 0.10 * max(arch, 0.0), 0.05, 0.25))

    # market maker fraction: fixed small slice
    mm_frac = 0.05

    # noise fills the remainder
    noise_frac = float(np.clip(
        1.0 - informed_frac - momentum_frac - mr_frac - mm_frac,
        0.1, 0.7,
    ))

    # re-normalise to ensure fractions sum to exactly 1
    total = informed_frac + momentum_frac + mr_frac + mm_frac + noise_frac
    informed_frac /= total
    momentum_frac /= total
    mr_frac /= total
    mm_frac /= total
    noise_frac /= total

    # calibrate threshold for momentum: roughly 2 * daily vol
    momentum_threshold = float(np.clip(2.0 * sigma, 0.005, 0.05))

    # mean-reversion z-score threshold: adjusted for tail thickness
    mr_z = float(np.clip(1.5 + 0.5 * tail, 1.5, 3.5))

    # noise agent sigma calibrated to empirical vol
    noise_sigma = float(np.clip(sigma * 500, 10.0, 200.0))

    logger.info(
        "Calibration: noise=%.3f mom=%.3f mr=%.3f mm=%.3f inf=%.3f",
        noise_frac, momentum_frac, mr_frac, mm_frac, informed_frac,
    )

    return AgentPopulationConfig(
        noise_fraction=noise_frac,
        momentum_fraction=momentum_frac,
        mean_reversion_fraction=mr_frac,
        informed_fraction=informed_frac,
        market_maker_fraction=mm_frac,
        momentum_threshold=momentum_threshold,
        mean_reversion_z=mr_z,
        noise_sigma=noise_sigma,
    )


def build_market_from_config(
    config: AgentPopulationConfig,
    initial_price: float = 100.0,
    total_agents: int = 50,
    price_impact: float = 1e-4,
    rng: Optional[np.random.Generator] = None,
) -> AgentBasedMarket:
    """
    Instantiate an AgentBasedMarket populated according to a calibrated config.

    Parameters
    --------
    config : AgentPopulationConfig
        Output of fit_to_empirical().
    initial_price : float
        Starting price for the simulation.
    total_agents : int
        Total number of agents to populate.
    price_impact : float
        Price impact coefficient.
    rng : optional random generator

    Returns
    -----
    AgentBasedMarket
    """
    if rng is None:
        rng = np.random.default_rng()

    market = AgentBasedMarket(
        initial_price=initial_price,
        price_impact=price_impact,
        rng=rng,
    )

    counts = {
        "noise": max(1, round(config.noise_fraction * total_agents)),
        "momentum": max(1, round(config.momentum_fraction * total_agents)),
        "mr": max(1, round(config.mean_reversion_fraction * total_agents)),
        "mm": max(1, round(config.market_maker_fraction * total_agents)),
        "informed": max(1, round(config.informed_fraction * total_agents)),
    }

    for i in range(counts["noise"]):
        market.add_agent(NoiseAgent(
            agent_id=f"noise_{i}",
            sigma_size=config.noise_sigma,
            rng=np.random.default_rng(rng.integers(0, 2**32)),
        ))

    for i in range(counts["momentum"]):
        market.add_agent(MomentumAgent(
            agent_id=f"momentum_{i}",
            threshold=config.momentum_threshold,
        ))

    for i in range(counts["mr"]):
        market.add_agent(MeanReversionAgent(
            agent_id=f"mr_{i}",
            z_threshold=config.mean_reversion_z,
        ))

    for i in range(counts["mm"]):
        market.add_agent(MarketMakerAgent(agent_id=f"mm_{i}"))

    for i in range(counts["informed"]):
        market.add_agent(InformedTrader(
            agent_id=f"informed_{i}",
            true_value=initial_price,
            rng=np.random.default_rng(rng.integers(0, 2**32)),
        ))

    return market
