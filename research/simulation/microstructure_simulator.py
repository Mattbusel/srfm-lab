"""
research/simulation/microstructure_simulator.py

Limit order book (LOB) simulator, synthetic tick data generator, and
spread dynamics simulator for SRFM microstructure research.

Components:
  - LOBSimulator: price-level LOB with limit/market/cancel order handling
  - TickDataGenerator: synthetic tick stream from LOB dynamics
  - SpreadDynamicsSimulator: adverse selection and liquidity crisis scenarios
"""

from __future__ import annotations

import logging
import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LimitOrder:
    """A resting limit order in the book."""
    order_id: str
    side: str            # "bid" or "ask"
    price: float
    qty: float
    timestamp: float     # seconds since simulation start


@dataclass
class Fill:
    """A fill event from order matching."""
    aggressor_id: str    # "market" for market orders
    passive_id: str      # order_id of resting order
    side: str            # side of the aggressor
    price: float
    qty: float
    timestamp: float


@dataclass
class Tick:
    """A single market data tick."""
    timestamp: float
    price: float
    qty: float
    side: str             # "buy" or "sell"
    tick_type: str        # "trade", "quote_bid", "quote_ask", "cancel"
    bid: Optional[float]
    ask: Optional[float]
    spread_bps: Optional[float]


# ---------------------------------------------------------------------------
# LOBSimulator
# ---------------------------------------------------------------------------

class LOBSimulator:
    """
    Limit Order Book simulator with price-level aggregation.

    Maintains sorted bid and ask queues. Market orders sweep the book
    greedily; limit orders rest at their specified price level.
    """

    def __init__(self, tick_size: float = 0.01) -> None:
        self.tick_size = tick_size
        self._clock: float = 0.0

        # bids: dict[price -> list[LimitOrder]], prices sorted descending
        self._bids: Dict[float, List[LimitOrder]] = {}
        # asks: dict[price -> list[LimitOrder]], prices sorted ascending
        self._asks: Dict[float, List[LimitOrder]] = {}
        # all active orders by id
        self._orders: Dict[str, LimitOrder] = {}

        self._fills: List[Fill] = []
        self._n_trades: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _round_price(self, price: float) -> float:
        """Round price to nearest tick."""
        return round(round(price / self.tick_size) * self.tick_size, 10)

    def _sorted_bid_prices(self) -> List[float]:
        """Bid prices in descending order (best bid first)."""
        return sorted(self._bids.keys(), reverse=True)

    def _sorted_ask_prices(self) -> List[float]:
        """Ask prices in ascending order (best ask first)."""
        return sorted(self._asks.keys())

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def place_limit_order(
        self, side: str, price: float, qty: float
    ) -> str:
        """
        Place a resting limit order.

        Parameters
        # --------
        side : str
            "bid" or "ask"
        price : float
            Limit price.
        qty : float
            Order quantity (must be positive).

        Returns
        # -----
        str
            Unique order_id.
        """
        if side not in ("bid", "ask"):
            raise ValueError(f"side must be 'bid' or 'ask', got '{side}'")
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")

        price = self._round_price(price)
        order_id = str(uuid.uuid4())
        order = LimitOrder(
            order_id=order_id,
            side=side,
            price=price,
            qty=qty,
            timestamp=self._clock,
        )
        self._orders[order_id] = order

        book = self._bids if side == "bid" else self._asks
        if price not in book:
            book[price] = []
        book[price].append(order)

        logger.debug("Limit %s %s @ %.4f x %.2f id=%s", side, side, price, qty, order_id)
        return order_id

    def place_market_order(self, side: str, qty: float) -> List[Fill]:
        """
        Execute a market order against the resting book.

        Parameters
        # --------
        side : str
            "buy" or "sell"
        qty : float
            Quantity to fill (must be positive).

        Returns
        # -----
        List[Fill]
            All fills generated. May be partial if book is thin.
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")

        fills: List[Fill] = []
        remaining = qty

        if side == "buy":
            price_levels = self._sorted_ask_prices()
            book = self._asks
        else:
            price_levels = self._sorted_bid_prices()
            book = self._bids

        for level_price in price_levels:
            if remaining <= 1e-9:
                break
            if level_price not in book:
                continue

            level_orders = book[level_price]
            i = 0
            while i < len(level_orders) and remaining > 1e-9:
                resting = level_orders[i]
                if resting.qty <= 0:
                    i += 1
                    continue

                fill_qty = min(resting.qty, remaining)
                fill = Fill(
                    aggressor_id="market",
                    passive_id=resting.order_id,
                    side=side,
                    price=resting.price,
                    qty=fill_qty,
                    timestamp=self._clock,
                )
                fills.append(fill)
                self._fills.append(fill)

                resting.qty -= fill_qty
                remaining -= fill_qty
                self._n_trades += 1

                if resting.qty < 1e-9:
                    # order fully filled; remove from active orders
                    self._orders.pop(resting.order_id, None)
                    level_orders.pop(i)
                else:
                    i += 1

            # clean up empty price level
            if not level_orders:
                book.pop(level_price, None)

        if remaining > 1e-9:
            logger.debug(
                "Market %s partially filled; %.4f shares unfilled", side, remaining
            )

        return fills

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a resting limit order by id.

        Returns
        # -----
        bool
            True if the order was found and cancelled.
        """
        order = self._orders.pop(order_id, None)
        if order is None:
            return False

        book = self._bids if order.side == "bid" else self._asks
        level = book.get(order.price, [])
        book[order.price] = [o for o in level if o.order_id != order_id]
        if not book[order.price]:
            book.pop(order.price, None)

        logger.debug("Cancelled order %s", order_id)
        return True

    def best_bid(self) -> Optional[float]:
        """Return the best (highest) bid price, or None if book is empty."""
        prices = self._sorted_bid_prices()
        return prices[0] if prices else None

    def best_ask(self) -> Optional[float]:
        """Return the best (lowest) ask price, or None if book is empty."""
        prices = self._sorted_ask_prices()
        return prices[0] if prices else None

    def mid_price(self) -> Optional[float]:
        """Return the mid-price, or None if one side is empty."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def spread_bps(self) -> Optional[float]:
        """Return the bid-ask spread in basis points."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None or bid <= 0:
            return None
        return (ask - bid) / bid * 10_000.0

    def book_depth(
        self, n_levels: int = 10
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Return top-N levels of bid and ask book.

        Returns
        # -----
        Tuple of (bids, asks) where each is a list of (price, total_size) tuples.
        """
        bid_prices = self._sorted_bid_prices()[:n_levels]
        ask_prices = self._sorted_ask_prices()[:n_levels]

        bids = []
        for p in bid_prices:
            total_qty = sum(o.qty for o in self._bids.get(p, []))
            if total_qty > 0:
                bids.append((p, total_qty))

        asks = []
        for p in ask_prices:
            total_qty = sum(o.qty for o in self._asks.get(p, []))
            if total_qty > 0:
                asks.append((p, total_qty))

        return bids, asks

    def advance_clock(self, dt: float) -> None:
        """Advance the internal simulation clock by dt seconds."""
        self._clock += dt

    def n_active_orders(self) -> int:
        """Number of currently resting orders."""
        return len(self._orders)

    def reset(self) -> None:
        """Clear all state."""
        self._bids.clear()
        self._asks.clear()
        self._orders.clear()
        self._fills.clear()
        self._clock = 0.0
        self._n_trades = 0

    def _seed_book(
        self,
        mid: float,
        n_levels: int = 5,
        level_spacing: float = 0.01,
        size_per_level: float = 100.0,
    ) -> None:
        """
        Populate both sides of the book with n_levels of limit orders
        around `mid`. Used by generators to initialise a non-empty book.
        """
        for i in range(1, n_levels + 1):
            bid_px = self._round_price(mid - i * level_spacing)
            ask_px = self._round_price(mid + i * level_spacing)
            size = size_per_level * (1.0 + 0.5 * (n_levels - i) / n_levels)
            if bid_px > 0:
                self.place_limit_order("bid", bid_px, size)
            self.place_limit_order("ask", ask_px, size)


# ---------------------------------------------------------------------------
# TickDataGenerator
# ---------------------------------------------------------------------------

class TickDataGenerator:
    """
    Generates a realistic synthetic tick stream by driving a LOBSimulator
    with a Poisson-arrival order process.

    Order type mix (configurable):
      - 60% limit orders  # resting orders near mid
      - 30% market orders # immediate execution
      - 10% cancel orders # random cancellation of resting orders

    Price levels for limit orders are drawn from a normal distribution
    centred on the current mid with std proportional to spread.
    """

    def __init__(
        self,
        initial_price: float = 100.0,
        tick_size: float = 0.01,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.initial_price = initial_price
        self.tick_size = tick_size
        self._rng = rng if rng is not None else np.random.default_rng()
        self._lob = LOBSimulator(tick_size=tick_size)

    def generate(
        self,
        n_ticks: int,
        volatility: float = 0.0002,
        arrival_rate: float = 10.0,
        limit_fraction: float = 0.60,
        market_fraction: float = 0.30,
    ) -> List[Tick]:
        """
        Generate a synthetic tick stream.

        Parameters
        # --------
        n_ticks : int
            Number of ticks to generate.
        volatility : float
            Per-tick price volatility (std of log-return per event).
        arrival_rate : float
            Mean order arrivals per second (Poisson rate).
        limit_fraction : float
            Fraction of arrivals that are limit orders.
        market_fraction : float
            Fraction of arrivals that are market orders.
            Remaining (1 - limit - market) are cancel orders.

        Returns
        # -----
        List[Tick]
        """
        if limit_fraction + market_fraction > 1.0:
            raise ValueError("limit_fraction + market_fraction must be <= 1.0")

        cancel_fraction = 1.0 - limit_fraction - market_fraction

        self._lob.reset()
        self._lob._seed_book(
            mid=self.initial_price,
            n_levels=5,
            level_spacing=self.initial_price * 0.0005,
            size_per_level=200.0,
        )

        ticks: List[Tick] = []
        current_mid = self.initial_price
        clock = 0.0

        for _ in range(n_ticks):
            # inter-arrival time: exponential with mean 1/arrival_rate
            dt = float(self._rng.exponential(1.0 / arrival_rate))
            clock += dt
            self._lob.advance_clock(dt)

            # mid drifts by small random walk
            shock = self._rng.normal(0.0, volatility)
            current_mid *= math.exp(shock)

            u = self._rng.random()

            if u < limit_fraction:
                tick = self._gen_limit_order(clock, current_mid)
            elif u < limit_fraction + market_fraction:
                tick = self._gen_market_order(clock, current_mid)
            else:
                tick = self._gen_cancel(clock)

            if tick is not None:
                ticks.append(tick)

            # re-seed book if thin
            mid_now = self._lob.mid_price()
            if mid_now is None or self._lob.n_active_orders() < 4:
                self._lob._seed_book(
                    mid=current_mid,
                    n_levels=3,
                    level_spacing=current_mid * 0.0005,
                    size_per_level=100.0,
                )

        return ticks

    # ------------------------------------------------------------------
    # Internal order generators
    # ------------------------------------------------------------------
    def _gen_limit_order(self, clock: float, mid: float) -> Optional[Tick]:
        """Place a limit order near the mid and return a quote tick."""
        half_spread = mid * 0.0005
        side = "bid" if self._rng.random() < 0.5 else "ask"

        if side == "bid":
            offset = abs(self._rng.normal(half_spread, half_spread))
            price = max(self.tick_size, mid - offset)
        else:
            offset = abs(self._rng.normal(half_spread, half_spread))
            price = mid + offset

        qty = abs(self._rng.normal(100.0, 50.0))
        qty = max(1.0, qty)

        try:
            self._lob.place_limit_order(side, price, qty)
        except ValueError:
            return None

        bid = self._lob.best_bid()
        ask = self._lob.best_ask()
        spd = self._lob.spread_bps()

        tick_type = "quote_bid" if side == "bid" else "quote_ask"
        return Tick(
            timestamp=clock,
            price=price,
            qty=qty,
            side=side,
            tick_type=tick_type,
            bid=bid,
            ask=ask,
            spread_bps=spd,
        )

    def _gen_market_order(self, clock: float, mid: float) -> Optional[Tick]:
        """Execute a small market order and return a trade tick."""
        side = "buy" if self._rng.random() < 0.5 else "sell"
        qty = abs(self._rng.normal(50.0, 30.0))
        qty = max(1.0, qty)

        try:
            fills = self._lob.place_market_order(side, qty)
        except ValueError:
            return None

        if not fills:
            return None

        # volume-weighted average fill price
        total_qty = sum(f.qty for f in fills)
        vwap = sum(f.price * f.qty for f in fills) / (total_qty + 1e-12)

        bid = self._lob.best_bid()
        ask = self._lob.best_ask()
        spd = self._lob.spread_bps()

        return Tick(
            timestamp=clock,
            price=vwap,
            qty=total_qty,
            side=side,
            tick_type="trade",
            bid=bid,
            ask=ask,
            spread_bps=spd,
        )

    def _gen_cancel(self, clock: float) -> Optional[Tick]:
        """Cancel a random resting order and return a cancel tick."""
        active_ids = list(self._lob._orders.keys())
        if not active_ids:
            return None

        target_id = self._rng.choice(active_ids)
        order = self._lob._orders.get(target_id)
        if order is None:
            return None

        price = order.price
        qty = order.qty
        side = order.side
        self._lob.cancel_order(target_id)

        bid = self._lob.best_bid()
        ask = self._lob.best_ask()
        spd = self._lob.spread_bps()

        return Tick(
            timestamp=clock,
            price=price,
            qty=qty,
            side=side,
            tick_type="cancel",
            bid=bid,
            ask=ask,
            spread_bps=spd,
        )


# ---------------------------------------------------------------------------
# SpreadDynamicsSimulator
# ---------------------------------------------------------------------------

class SpreadDynamicsSimulator:
    """
    Simulates bid-ask spread dynamics under stress scenarios.

    Models:
      - Adverse selection: informed traders arrive; spreads widen to
        compensate market makers for expected losses.
      - Liquidity crisis: sudden spread blow-out at a specified step,
        followed by slow mean-reversion.
    """

    def __init__(
        self,
        base_spread_bps: float = 5.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.base_spread_bps = base_spread_bps
        self._rng = rng if rng is not None else np.random.default_rng()

    def simulate_adverse_selection_event(
        self,
        n_steps: int,
        informed_pct: float = 0.30,
        impact_per_informed: float = 20.0,
        mean_reversion_speed: float = 0.05,
    ) -> pd.DataFrame:
        """
        Simulate spread widening caused by a sustained adverse-selection event.

        The spread follows an OU process with a higher mean while informed
        traders are active. Informed fraction decays exponentially from
        `informed_pct` back toward zero.

        Parameters
        # --------
        n_steps : int
            Number of simulation steps.
        informed_pct : float
            Peak fraction of order flow from informed traders (0-1).
        impact_per_informed : float
            Additional spread (bps) per unit of informed fraction.
        mean_reversion_speed : float
            Speed of spread mean-reversion per step.

        Returns
        # -----
        pd.DataFrame with columns: step, spread_bps, informed_fraction,
            mid_price, volume.
        """
        if informed_pct < 0 or informed_pct > 1:
            raise ValueError("informed_pct must be in [0, 1]")

        records = []
        spread = self.base_spread_bps
        mid = 100.0
        decay_rate = 3.0 / n_steps    # informed fraction decays to ~5% of peak

        for step in range(n_steps):
            inf_frac = informed_pct * math.exp(-decay_rate * step)
            target_spread = self.base_spread_bps + impact_per_informed * inf_frac
            # OU mean-reversion with noise
            noise = self._rng.normal(0.0, 0.5)
            spread += mean_reversion_speed * (target_spread - spread) + noise
            spread = max(0.5, spread)

            # mid price: informed traders push price, noise otherwise
            price_shock = self._rng.normal(0.0, 0.001) * (1.0 + 3.0 * inf_frac)
            mid *= math.exp(price_shock)

            volume = abs(self._rng.normal(1000.0, 300.0)) * (1.0 + inf_frac)

            records.append({
                "step": step,
                "spread_bps": spread,
                "informed_fraction": inf_frac,
                "mid_price": mid,
                "volume": volume,
            })

        return pd.DataFrame(records)

    def simulate_liquidity_crisis(
        self,
        n_steps: int,
        crisis_onset: int,
        crisis_spread_bps: float = 50.0,
        recovery_speed: float = 0.02,
        crisis_vol_multiplier: float = 3.0,
    ) -> pd.DataFrame:
        """
        Simulate a sudden liquidity crisis where spread blows out at
        `crisis_onset` and then slowly recovers.

        Parameters
        # --------
        n_steps : int
            Total simulation steps.
        crisis_onset : int
            Step at which the crisis begins.
        crisis_spread_bps : float
            Spread level immediately after crisis onset.
        recovery_speed : float
            OU mean-reversion speed back to base spread after crisis.
        crisis_vol_multiplier : float
            Multiplier on price volatility during and after crisis.

        Returns
        # -----
        pd.DataFrame with columns: step, spread_bps, crisis_active,
            mid_price, volume, volatility.
        """
        if crisis_onset < 0 or crisis_onset >= n_steps:
            raise ValueError(
                f"crisis_onset must be in [0, n_steps), got {crisis_onset}"
            )

        records = []
        spread = self.base_spread_bps
        mid = 100.0
        vol_multiplier = 1.0

        for step in range(n_steps):
            crisis_active = step >= crisis_onset

            if step == crisis_onset:
                # instantaneous spread jump
                spread = crisis_spread_bps
                vol_multiplier = crisis_vol_multiplier

            if crisis_active:
                # slow mean-reversion back to base spread
                target = self.base_spread_bps
                noise = self._rng.normal(0.0, 1.0)
                spread += recovery_speed * (target - spread) + noise
                spread = max(self.base_spread_bps * 0.8, spread)
                # vol also decays
                vol_multiplier = max(
                    1.0,
                    vol_multiplier - recovery_speed * (vol_multiplier - 1.0),
                )
            else:
                # pre-crisis: mild OU noise around base
                noise = self._rng.normal(0.0, 0.3)
                spread += 0.1 * (self.base_spread_bps - spread) + noise
                spread = max(0.5, spread)

            current_vol = 0.001 * vol_multiplier
            price_shock = self._rng.normal(0.0, current_vol)
            mid *= math.exp(price_shock)

            volume = max(0.0, self._rng.normal(
                800.0 / vol_multiplier, 200.0
            ))

            records.append({
                "step": step,
                "spread_bps": spread,
                "crisis_active": crisis_active,
                "mid_price": mid,
                "volume": volume,
                "volatility": current_vol,
            })

        return pd.DataFrame(records)
