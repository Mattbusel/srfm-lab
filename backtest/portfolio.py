"""
portfolio.py -- Portfolio accounting and position management for LARSA.

Implements NaivePortfolio (equal-weight baseline) and LARSAPortfolio
(fractional target weights, ramp logic, min-hold enforcement) with
full P&L accounting, rebalance scheduling, and cross-margin simulation.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .engine import (
    Direction,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderType,
    PositionState,
    SignalEvent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rebalance Scheduler
# ---------------------------------------------------------------------------

class RebalanceScheduler:
    """
    Enforces minimum rebalance cadence. For LARSA this is every 15-min bar.
    Also enforces minimum hold time to prevent over-trading.
    """

    def __init__(
        self,
        rebalance_freq_bars: int = 1,   # rebalance every N bars
        min_hold_bars: int = 4,         # minimum 4 bars (1 hour) hold
        cooldown_bars: int = 2,         # bars between repeated signals
    ):
        self.rebalance_freq = rebalance_freq_bars
        self.min_hold_bars = min_hold_bars
        self.cooldown_bars = cooldown_bars
        self._bar_count: int = 0
        self._last_rebalance: int = 0
        self._last_trade: Dict[str, int] = {}
        self._entry_bar: Dict[str, int] = {}

    def tick(self) -> None:
        self._bar_count += 1

    def can_rebalance(self) -> bool:
        return (self._bar_count - self._last_rebalance) >= self.rebalance_freq

    def mark_rebalanced(self) -> None:
        self._last_rebalance = self._bar_count

    def can_trade(self, symbol: str) -> bool:
        """True if cooldown has expired for the symbol."""
        last = self._last_trade.get(symbol, -999)
        return (self._bar_count - last) >= self.cooldown_bars

    def mark_traded(self, symbol: str) -> None:
        self._last_trade[symbol] = self._bar_count

    def can_exit(self, symbol: str) -> bool:
        """True if min-hold period has expired."""
        entry = self._entry_bar.get(symbol, -999)
        return (self._bar_count - entry) >= self.min_hold_bars

    def mark_entry(self, symbol: str) -> None:
        self._entry_bar[symbol] = self._bar_count

    @property
    def current_bar(self) -> int:
        return self._bar_count


# ---------------------------------------------------------------------------
# Margin Simulator
# ---------------------------------------------------------------------------

class MarginSimulator:
    """
    Cross-margin simulation for leveraged crypto trading.

    Tracks:
      - Initial margin requirement
      - Maintenance margin (liquidation threshold)
      - Available margin and margin utilization
    """

    def __init__(
        self,
        initial_margin_rate: float = 0.10,       # 10x leverage
        maintenance_margin_rate: float = 0.05,   # 5% maintenance
        margin_call_buffer: float = 0.02,
    ):
        self.init_margin_rate = initial_margin_rate
        self.maint_margin_rate = maintenance_margin_rate
        self.margin_call_buffer = margin_call_buffer
        self._total_equity: float = 0.0
        self._used_margin: float = 0.0
        self._positions: Dict[str, float] = {}  # symbol -> notional

    def update_equity(self, equity: float) -> None:
        self._total_equity = equity

    def compute_margin_required(self, notional: float) -> float:
        return notional * self.init_margin_rate

    def compute_maintenance_margin(self, notional: float) -> float:
        return notional * self.maint_margin_rate

    def add_position(self, symbol: str, notional: float) -> bool:
        """
        Attempt to add a position. Returns False if margin is insufficient.
        """
        required = self.compute_margin_required(abs(notional))
        if required > self.available_margin:
            logger.warning(
                "Insufficient margin for %s: required=%.2f, available=%.2f",
                symbol, required, self.available_margin,
            )
            return False
        self._positions[symbol] = notional
        self._update_used_margin()
        return True

    def remove_position(self, symbol: str) -> None:
        self._positions.pop(symbol, None)
        self._update_used_margin()

    def _update_used_margin(self) -> None:
        self._used_margin = sum(
            self.compute_margin_required(abs(n)) for n in self._positions.values()
        )

    @property
    def available_margin(self) -> float:
        return max(0.0, self._total_equity - self._used_margin)

    @property
    def margin_utilization(self) -> float:
        if self._total_equity <= 0:
            return 0.0
        return self._used_margin / self._total_equity

    @property
    def is_margin_call(self) -> bool:
        """True if equity falls below maintenance margin on total exposure."""
        total_notional = sum(abs(n) for n in self._positions.values())
        maint = self.compute_maintenance_margin(total_notional)
        return self._total_equity < maint + self.margin_call_buffer * total_notional

    def get_max_position_notional(self) -> float:
        return self.available_margin / (self.init_margin_rate + 1e-12)


# ---------------------------------------------------------------------------
# Base Portfolio class
# ---------------------------------------------------------------------------

class BasePortfolio:
    """Abstract base for portfolio implementations."""

    def __init__(self, initial_capital: float, symbols: List[str]):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.symbols = symbols
        self.positions: Dict[str, PositionState] = {
            s: PositionState(symbol=s) for s in symbols
        }
        self._current_prices: Dict[str, float] = {s: 0.0 for s in symbols}
        self._orders_issued: List[OrderEvent] = []
        self._fills_received: List[FillEvent] = []
        self.equity_snapshots: List[Tuple[pd.Timestamp, float]] = []

    def on_market_event(self, event: MarketEvent) -> None:
        self._current_prices[event.symbol] = event.close
        for sym, pos in self.positions.items():
            p = self._current_prices.get(sym, pos.avg_price)
            pos.update_market(p)

    def on_signal_event(self, event: SignalEvent) -> Optional[List[OrderEvent]]:
        raise NotImplementedError

    def on_order_event(self, event: OrderEvent) -> None:
        self._orders_issued.append(event)

    def on_fill_event(self, event: FillEvent) -> None:
        self._fills_received.append(event)
        self._apply_fill(event)

    def _apply_fill(self, event: FillEvent) -> None:
        sym = event.symbol
        if sym not in self.positions:
            self.positions[sym] = PositionState(symbol=sym)
        self.positions[sym].apply_fill(event.quantity, event.fill_price)
        cash_delta = -event.quantity * event.fill_price - event.commission - event.slippage
        self.cash += cash_delta

    @property
    def equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions.values())

    @property
    def net_exposure(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def leverage(self) -> float:
        eq = self.equity
        return self.gross_exposure / eq if eq > 0 else 0.0

    def get_weights(self) -> Dict[str, float]:
        eq = self.equity
        if eq <= 0:
            return {}
        return {sym: pos.market_value / eq for sym, pos in self.positions.items()}

    def snapshot(self, timestamp: pd.Timestamp) -> Dict[str, Any]:
        return {
            "timestamp": timestamp,
            "equity": self.equity,
            "cash": self.cash,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "leverage": self.leverage,
            "weights": self.get_weights(),
        }

    def _make_order(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        quantity: float,
        direction: Direction,
        price: float = 0.0,
        order_type: OrderType = OrderType.MARKET,
    ) -> OrderEvent:
        return OrderEvent(
            event_type=EventType.ORDER,
            timestamp=timestamp,
            symbol=symbol,
            order_type=order_type,
            quantity=quantity,
            price=price,
            direction=direction,
            order_id=str(uuid.uuid4())[:8],
        )


# ---------------------------------------------------------------------------
# NaivePortfolio: equal-weight baseline
# ---------------------------------------------------------------------------

class NaivePortfolio(BasePortfolio):
    """
    Baseline portfolio that allocates equal weight to each long signal
    and ignores short signals. Used for benchmarking LARSA.

    Rebalances every bar. No min-hold enforcement.
    """

    def __init__(
        self,
        initial_capital: float,
        symbols: List[str],
        weight_per_signal: float = 0.25,
        max_positions: int = 4,
    ):
        super().__init__(initial_capital, symbols)
        self.weight_per_signal = weight_per_signal
        self.max_positions = max_positions
        self._active_longs: set = set()
        self._bar_count: int = 0

    def on_market_event(self, event: MarketEvent) -> None:
        super().on_market_event(event)
        self._bar_count += 1
        self.equity_snapshots.append((event.timestamp, self.equity))

    def on_signal_event(self, event: SignalEvent) -> Optional[List[OrderEvent]]:
        orders = []
        sym = event.symbol
        price = self._current_prices.get(sym, 0.0)
        if price <= 0:
            return None

        eq = self.equity
        pos = self.positions.get(sym, PositionState(symbol=sym))

        if event.direction == Direction.LONG:
            if sym not in self._active_longs and len(self._active_longs) < self.max_positions:
                target_notional = eq * self.weight_per_signal
                target_qty = target_notional / price
                delta_qty = target_qty - pos.quantity
                if abs(delta_qty) > 1e-6:
                    self._active_longs.add(sym)
                    orders.append(self._make_order(event.timestamp, sym, delta_qty, Direction.LONG, price))

        elif event.direction in (Direction.EXIT, Direction.FLAT, Direction.SHORT):
            if sym in self._active_longs and abs(pos.quantity) > 1e-6:
                self._active_longs.discard(sym)
                orders.append(self._make_order(event.timestamp, sym, -pos.quantity, Direction.EXIT, price))

        return orders if orders else None


# ---------------------------------------------------------------------------
# LARSAPortfolio: full LARSA position management
# ---------------------------------------------------------------------------

class LARSAPortfolio(BasePortfolio):
    """
    LARSA portfolio with:
      - Fractional target weights from signal strength
      - Ramp-up/ramp-down logic (linearly transitions to target weight)
      - Min-hold enforcement (no premature exits)
      - Cross-margin simulation
      - Position size capping
    """

    def __init__(
        self,
        initial_capital: float,
        symbols: List[str],
        max_position_weight: float = 0.40,
        max_leverage: float = 2.0,
        ramp_bars: int = 3,           # bars to ramp into a full position
        min_hold_bars: int = 4,
        cooldown_bars: int = 2,
        use_margin: bool = False,
        rebalance_freq: int = 1,
    ):
        super().__init__(initial_capital, symbols)
        self.max_position_weight = max_position_weight
        self.max_leverage = max_leverage
        self.ramp_bars = ramp_bars
        self.use_margin = use_margin

        self.scheduler = RebalanceScheduler(
            rebalance_freq_bars=rebalance_freq,
            min_hold_bars=min_hold_bars,
            cooldown_bars=cooldown_bars,
        )

        if use_margin:
            self.margin_sim = MarginSimulator()
        else:
            self.margin_sim = None

        # Target weights from last signal
        self._target_weights: Dict[str, float] = {s: 0.0 for s in symbols}
        # Current ramp stage per symbol (0 to ramp_bars)
        self._ramp_stage: Dict[str, int] = {s: 0 for s in symbols}
        self._entry_ts: Dict[str, Optional[pd.Timestamp]] = {s: None for s in symbols}
        self._last_direction: Dict[str, Direction] = {}
        self._pending_orders: List[OrderEvent] = []
        self._bar_count: int = 0

    def on_market_event(self, event: MarketEvent) -> Optional[List[OrderEvent]]:
        super().on_market_event(event)
        self._bar_count += 1
        self.scheduler.tick()
        self.equity_snapshots.append((event.timestamp, self.equity))

        if self.margin_sim:
            self.margin_sim.update_equity(self.equity)

        # Execute any ramp orders
        ramp_orders = self._process_ramp(event)
        return ramp_orders if ramp_orders else None

    def on_signal_event(self, event: SignalEvent) -> Optional[List[OrderEvent]]:
        """
        Accept new signal. Updates target weight and starts ramp logic.
        Actual orders are emitted on the next market event.
        """
        sym = event.symbol
        if not self.scheduler.can_trade(sym):
            logger.debug("Cooldown active for %s, skipping signal", sym)
            return None

        # Check min-hold for exit
        if event.direction in (Direction.EXIT, Direction.FLAT):
            if not self.scheduler.can_exit(sym):
                logger.debug("Min-hold active for %s, blocking exit", sym)
                return None

        # Update target weight
        if event.direction == Direction.LONG:
            self._target_weights[sym] = abs(event.target_weight or event.strength * self.max_position_weight)
        elif event.direction == Direction.SHORT:
            self._target_weights[sym] = -abs(event.target_weight or event.strength * self.max_position_weight)
        elif event.direction in (Direction.EXIT, Direction.FLAT):
            self._target_weights[sym] = 0.0

        self._last_direction[sym] = event.direction
        self._ramp_stage[sym] = 0
        self.scheduler.mark_traded(sym)
        if event.direction not in (Direction.EXIT, Direction.FLAT):
            self.scheduler.mark_entry(sym)
            self._entry_ts[sym] = event.timestamp

        return None  # orders issued by _process_ramp

    def _process_ramp(self, event: MarketEvent) -> List[OrderEvent]:
        """
        For each symbol that has an active ramp, compute the incremental
        order to reach the next ramp fraction of the target weight.
        """
        orders = []
        if not self.scheduler.can_rebalance():
            return orders
        self.scheduler.mark_rebalanced()

        eq = self.equity
        if eq <= 0:
            return orders

        for sym, target_w in self._target_weights.items():
            price = self._current_prices.get(sym, 0.0)
            if price <= 0:
                continue

            pos = self.positions.get(sym, PositionState(symbol=sym))
            current_w = pos.market_value / eq

            # Ramp fraction: linearly increase from 0 to target_w over ramp_bars
            stage = self._ramp_stage.get(sym, self.ramp_bars)
            if stage >= self.ramp_bars:
                # Fully ramped: just track target drift
                desired_w = target_w
            else:
                frac = (stage + 1) / self.ramp_bars
                desired_w = target_w * frac
                self._ramp_stage[sym] = stage + 1

            # Cap by max weight and leverage constraints
            desired_w = np.clip(desired_w, -self.max_position_weight, self.max_position_weight)
            desired_w = self._apply_leverage_cap(sym, desired_w, eq)

            # Compute delta
            delta_w = desired_w - current_w
            delta_notional = delta_w * eq
            delta_qty = delta_notional / price

            if abs(delta_qty * price) < eq * 0.001:  # skip if < 0.1% of equity
                continue

            direction = Direction.LONG if delta_qty > 0 else Direction.SHORT
            if desired_w == 0 and abs(pos.quantity) < 1e-8:
                continue

            # Margin check
            if self.margin_sim and not self.margin_sim.add_position(sym, desired_w * eq):
                logger.warning("Margin blocked order for %s", sym)
                continue

            order = self._make_order(
                event.timestamp, sym, delta_qty, direction, price
            )
            orders.append(order)

        return orders

    def _apply_leverage_cap(self, sym: str, desired_w: float, equity: float) -> float:
        """Scale down weight to stay within max leverage."""
        # Current gross exposure plus new
        current_gross = self.gross_exposure
        current_w = self.positions.get(sym, PositionState(symbol=sym)).market_value / (equity + 1e-12)
        delta_exposure = abs(desired_w - current_w) * equity
        new_gross = current_gross + delta_exposure
        if new_gross / (equity + 1e-12) > self.max_leverage:
            # Scale back
            scale = (self.max_leverage * equity - current_gross) / (delta_exposure + 1e-12)
            scale = max(0.0, min(1.0, scale))
            desired_w = current_w + (desired_w - current_w) * scale
        return desired_w

    def on_fill_event(self, event: FillEvent) -> None:
        super().on_fill_event(event)

    def get_target_weights(self) -> Dict[str, float]:
        return dict(self._target_weights)

    def get_realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions.values())

    def get_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    def risk_report(self) -> Dict[str, Any]:
        """Return a snapshot of portfolio risk metrics."""
        eq = self.equity
        return {
            "equity": eq,
            "cash": self.cash,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "leverage": self.leverage,
            "realized_pnl": self.get_realized_pnl(),
            "unrealized_pnl": self.get_unrealized_pnl(),
            "margin_utilization": self.margin_sim.margin_utilization if self.margin_sim else None,
            "num_open_positions": sum(1 for p in self.positions.values() if abs(p.quantity) > 1e-8),
            "target_weights": self.get_target_weights(),
        }

    def to_equity_series(self) -> pd.Series:
        if not self.equity_snapshots:
            return pd.Series(dtype=float)
        ts = [t for t, _ in self.equity_snapshots]
        vals = [v for _, v in self.equity_snapshots]
        return pd.Series(vals, index=pd.DatetimeIndex(ts, name="timestamp"), name="equity")

    def force_flatten(self, timestamp: pd.Timestamp) -> List[OrderEvent]:
        """Immediately flatten all positions (end of backtest)."""
        orders = []
        for sym, pos in self.positions.items():
            if abs(pos.quantity) < 1e-8:
                continue
            price = self._current_prices.get(sym, pos.avg_price)
            qty = -pos.quantity
            direction = Direction.SHORT if qty < 0 else Direction.LONG
            orders.append(self._make_order(timestamp, sym, qty, direction, price))
            self._target_weights[sym] = 0.0
        return orders


# ---------------------------------------------------------------------------
# Trade Journal helper (used by portfolio for attribution)
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Single round-trip trade record."""
    symbol: str
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: float
    quantity: float
    direction: Direction
    gross_pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    entry_signal: str = ""
    exit_signal: str = ""
    bh_mass_at_entry: float = 0.0
    hurst_at_entry: float = 0.5
    regime_at_entry: str = ""

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.commission - self.slippage

    @property
    def hold_bars(self) -> int:
        if self.exit_time is None:
            return 0
        delta = self.exit_time - self.entry_time
        bars = int(delta.total_seconds() / 900)  # 900s = 15 min
        return max(bars, 0)

    @property
    def return_pct(self) -> float:
        notional = abs(self.entry_price * self.quantity)
        return self.net_pnl / notional if notional > 0 else 0.0
