"""
engine.py -- Event-driven backtesting engine for LARSA strategy.

Architecture: Events flow through a priority queue ordered by timestamp.
  MarketEvent -> Strategy -> SignalEvent -> Portfolio -> OrderEvent -> Execution -> FillEvent
"""

from __future__ import annotations

import heapq
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(Enum):
    MARKET = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()
    REBALANCE = auto()


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"
    FLAT = "FLAT"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    MOC = "MOC"  # market on close


@dataclass
class BaseEvent:
    """Base class for all backtest events."""
    event_type: EventType
    timestamp: pd.Timestamp
    priority: int = 0  # lower = higher priority

    def __lt__(self, other: "BaseEvent") -> bool:
        if self.timestamp == other.timestamp:
            return self.priority < other.priority
        return self.timestamp < other.timestamp

    def __le__(self, other: "BaseEvent") -> bool:
        return self == other or self < other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseEvent):
            return False
        return self.timestamp == other.timestamp and self.priority == other.priority


@dataclass
class MarketEvent(BaseEvent):
    """Carries OHLCV bar data for a single symbol/timeframe tick."""
    symbol: str = ""
    timeframe: str = "15m"
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    # Optional fields
    vwap: float = 0.0
    num_trades: int = 0

    def __post_init__(self):
        self.event_type = EventType.MARKET
        self.priority = 0  # markets process first

    @property
    def bar(self) -> Dict[str, float]:
        return {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
        }

    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0


@dataclass
class SignalEvent(BaseEvent):
    """Carries a directional signal from the strategy."""
    symbol: str = ""
    direction: Direction = Direction.FLAT
    strength: float = 0.0       # -1.0 to 1.0
    signal_id: str = ""
    # LARSA-specific metadata
    bh_mass: float = 0.0
    cf_cross: float = 0.0
    hurst: float = 0.5
    garch_vol: float = 0.0
    regime: str = "UNKNOWN"
    target_weight: float = 0.0  # fractional portfolio weight

    def __post_init__(self):
        self.event_type = EventType.SIGNAL
        self.priority = 1  # signals after market


@dataclass
class OrderEvent(BaseEvent):
    """Represents an order to be sent to the execution handler."""
    symbol: str = ""
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0       # positive = buy, negative = sell
    price: float = 0.0          # limit/stop price if applicable
    direction: Direction = Direction.LONG
    order_id: str = ""
    strategy_id: str = "LARSA"

    def __post_init__(self):
        self.event_type = EventType.ORDER
        self.priority = 2

    @property
    def is_buy(self) -> bool:
        return self.quantity > 0

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.price if self.price > 0 else 0.0


@dataclass
class FillEvent(BaseEvent):
    """Confirms execution of an order with actual fill details."""
    symbol: str = ""
    order_id: str = ""
    direction: Direction = Direction.LONG
    quantity: float = 0.0       # signed: positive = bought, negative = sold
    fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    exchange: str = "SIM"

    def __post_init__(self):
        self.event_type = EventType.FILL
        self.priority = 3

    @property
    def gross_value(self) -> float:
        return self.quantity * self.fill_price

    @property
    def net_cost(self) -> float:
        """Total cash impact (negative = cash outflow for a buy)."""
        return -self.quantity * self.fill_price - self.commission - abs(self.slippage)


# ---------------------------------------------------------------------------
# Event Queue
# ---------------------------------------------------------------------------

class EventQueue:
    """
    Priority queue for backtest events, ordered by (timestamp, priority).
    Uses heapq for O(log n) push/pop.
    """

    def __init__(self):
        self._heap: List[Tuple] = []
        self._counter = 0  # tie-breaker for equal-priority events

    def push(self, event: BaseEvent) -> None:
        entry = (event.timestamp, event.priority, self._counter, event)
        self._counter += 1
        heapq.heappush(self._heap, entry)

    def pop(self) -> BaseEvent:
        if self.empty():
            raise IndexError("EventQueue is empty")
        _, _, _, event = heapq.heappop(self._heap)
        return event

    def peek(self) -> Optional[BaseEvent]:
        if self.empty():
            return None
        _, _, _, event = self._heap[0]
        return event

    def empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)

    def clear(self) -> None:
        self._heap.clear()
        self._counter = 0


# ---------------------------------------------------------------------------
# Transaction cost model
# ---------------------------------------------------------------------------

class TransactionCostModel:
    """
    Models maker/taker fees and market-impact slippage.

    Slippage follows the square-root market impact model:
        impact = k * sigma * sqrt(Q / ADV)
    where k is an impact coefficient, sigma is daily vol,
    Q is order size and ADV is average daily volume.
    """

    # Default fee schedule (exchange-like)
    DEFAULT_FEES = {
        "crypto": {"maker": 0.0005, "taker": 0.0010},
        "equity": {"maker": 0.0002, "taker": 0.0005},
    }

    def __init__(
        self,
        maker_fee: float = 0.0005,
        taker_fee: float = 0.0010,
        impact_coeff: float = 0.1,
        asset_class: str = "crypto",
    ):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.impact_coeff = impact_coeff
        self.asset_class = asset_class

    def commission(self, notional: float, is_maker: bool = False) -> float:
        rate = self.maker_fee if is_maker else self.taker_fee
        return notional * rate

    def slippage(
        self,
        price: float,
        quantity: float,
        adv: float,
        daily_vol: float,
        is_buy: bool = True,
    ) -> float:
        """
        Square-root impact model.
        Returns slippage as a dollar amount (always positive cost).
        """
        if adv <= 0 or daily_vol <= 0:
            return 0.0
        participation = abs(quantity) / max(adv, 1e-8)
        impact_bps = self.impact_coeff * daily_vol * np.sqrt(participation)
        impact_bps = min(impact_bps, 0.05)  # cap at 5% of price
        direction = 1.0 if is_buy else -1.0
        slippage_per_share = price * impact_bps * direction
        return abs(slippage_per_share * quantity)

    def total_cost(
        self,
        price: float,
        quantity: float,
        adv: float = 1e6,
        daily_vol: float = 0.02,
        is_maker: bool = False,
        is_buy: bool = True,
    ) -> Tuple[float, float]:
        """Returns (commission, slippage) tuple."""
        notional = abs(price * quantity)
        comm = self.commission(notional, is_maker)
        slip = self.slippage(price, quantity, adv, daily_vol, is_buy)
        return comm, slip


# ---------------------------------------------------------------------------
# ADV tracker (rolling average daily volume)
# ---------------------------------------------------------------------------

class ADVTracker:
    """Tracks rolling average daily volume for slippage calculation."""

    def __init__(self, window: int = 20):
        self.window = window
        self._daily_vols: Dict[str, List[float]] = defaultdict(list)
        self._adv_cache: Dict[str, float] = {}

    def update(self, symbol: str, bar_volume: float, bars_per_day: int = 26) -> None:
        """Update with a 15-min bar volume. 26 bars per 6.5h trading day."""
        # Accumulate intraday volume and roll to daily
        key = symbol
        self._daily_vols[key].append(bar_volume)
        if len(self._daily_vols[key]) >= bars_per_day:
            day_vol = sum(self._daily_vols[key][-bars_per_day:])
            # Keep a separate daily list
            if not hasattr(self, "_daily_list"):
                self._daily_list: Dict[str, List[float]] = defaultdict(list)
            self._daily_list[key].append(day_vol)
            tail = self._daily_list[key][-self.window:]
            self._adv_cache[key] = float(np.mean(tail))

    def get_adv(self, symbol: str, fallback: float = 1e6) -> float:
        return self._adv_cache.get(symbol, fallback)


# ---------------------------------------------------------------------------
# Equity curve tracker
# ---------------------------------------------------------------------------

class EquityCurve:
    """Maintains the portfolio equity time series."""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self._timestamps: List[pd.Timestamp] = []
        self._equity: List[float] = []
        self._cash: List[float] = []
        self._drawdowns: List[float] = []
        self._peak: float = initial_capital

    def record(self, timestamp: pd.Timestamp, equity: float, cash: float) -> None:
        self._timestamps.append(timestamp)
        self._equity.append(equity)
        self._cash.append(cash)
        if equity > self._peak:
            self._peak = equity
        dd = (equity - self._peak) / self._peak if self._peak > 0 else 0.0
        self._drawdowns.append(dd)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "equity": self._equity,
                "cash": self._cash,
                "drawdown": self._drawdowns,
            },
            index=pd.DatetimeIndex(self._timestamps, name="timestamp"),
        )

    @property
    def current_equity(self) -> float:
        return self._equity[-1] if self._equity else self.initial_capital

    @property
    def peak_equity(self) -> float:
        return self._peak

    @property
    def max_drawdown(self) -> float:
        return min(self._drawdowns) if self._drawdowns else 0.0

    @property
    def total_return(self) -> float:
        if not self._equity:
            return 0.0
        return (self._equity[-1] - self.initial_capital) / self.initial_capital


# ---------------------------------------------------------------------------
# Portfolio state (positions)
# ---------------------------------------------------------------------------

@dataclass
class PositionState:
    """Tracks the state of a single position."""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    last_price: float = 0.0
    entry_time: Optional[pd.Timestamp] = None
    num_trades: int = 0

    def update_market(self, price: float) -> None:
        self.last_price = price
        self.unrealized_pnl = (price - self.avg_price) * self.quantity

    def apply_fill(self, quantity: float, price: float) -> float:
        """Apply a fill. Returns realized PnL from the fill."""
        realized = 0.0
        if self.quantity == 0:
            # Opening a new position
            self.avg_price = price
            self.quantity = quantity
        elif (self.quantity > 0 and quantity < 0) or (self.quantity < 0 and quantity > 0):
            # Closing (partially or fully)
            close_qty = min(abs(quantity), abs(self.quantity))
            realized = (price - self.avg_price) * close_qty * np.sign(self.quantity)
            self.realized_pnl += realized
            remaining = self.quantity + quantity
            if abs(remaining) < 1e-10:
                self.quantity = 0.0
                self.avg_price = 0.0
            elif np.sign(remaining) != np.sign(self.quantity):
                # Flip side
                self.avg_price = price
                self.quantity = remaining
            else:
                self.quantity = remaining
        else:
            # Adding to existing
            total_qty = self.quantity + quantity
            if abs(total_qty) > 1e-10:
                self.avg_price = (self.avg_price * abs(self.quantity) + price * abs(quantity)) / abs(total_qty)
            self.quantity = total_qty
        self.num_trades += 1
        return realized

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price if self.last_price > 0 else self.quantity * self.avg_price


# ---------------------------------------------------------------------------
# Main BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Event-driven backtesting engine.

    Lifecycle:
      1. load_data() -- populates the DataHandler
      2. run() -- processes events until the queue is empty
      3. Access results via equity_curve, trade_log, etc.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        cost_model: Optional[TransactionCostModel] = None,
        max_leverage: float = 3.0,
        verbose: bool = False,
    ):
        self.initial_capital = initial_capital
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        self.timeframes = timeframes or ["15m", "1h", "4h"]
        self.cost_model = cost_model or TransactionCostModel()
        self.max_leverage = max_leverage
        self.verbose = verbose

        # Core components (set by load_data / set_* methods)
        self._data_handler = None
        self._strategy = None
        self._portfolio = None
        self._execution = None

        # Runtime state
        self.event_queue = EventQueue()
        self.equity_curve = EquityCurve(initial_capital)
        self.cash = initial_capital
        self.positions: Dict[str, PositionState] = {}
        self.adv_tracker = ADVTracker()
        self.trade_log: List[Dict[str, Any]] = []
        self._bar_count = 0
        self._start_time: Optional[float] = None

        # Event handlers dispatch table
        self._handlers: Dict[EventType, List[Callable]] = {
            EventType.MARKET: [],
            EventType.SIGNAL: [],
            EventType.ORDER: [],
            EventType.FILL: [],
        }

        # Register default internal handlers
        self._register_defaults()

    def _register_defaults(self) -> None:
        self._handlers[EventType.MARKET].append(self._on_market)
        self._handlers[EventType.FILL].append(self._on_fill)

    def set_data_handler(self, handler) -> None:
        self._data_handler = handler
        self._handlers[EventType.MARKET].append(handler.on_market_event)

    def set_strategy(self, strategy) -> None:
        self._strategy = strategy
        self._handlers[EventType.MARKET].append(strategy.on_market_event)

    def set_portfolio(self, portfolio) -> None:
        self._portfolio = portfolio
        self._handlers[EventType.SIGNAL].append(portfolio.on_signal_event)
        self._handlers[EventType.FILL].append(portfolio.on_fill_event)
        self._handlers[EventType.ORDER].append(portfolio.on_order_event)

    def set_execution(self, execution) -> None:
        self._execution = execution
        self._handlers[EventType.ORDER].append(execution.on_order_event)

    def register_handler(self, event_type: EventType, handler: Callable) -> None:
        """Register an additional handler for an event type."""
        self._handlers[event_type].append(handler)

    def load_data(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> None:
        """
        Load market data into the engine. If data_handler is set,
        delegates there. Otherwise stores the provided dict.
        """
        if data is not None and self._data_handler is not None:
            self._data_handler.load(data)
        elif data is not None:
            self._raw_data = data
        logger.info("Data loaded for %d symbols", len(data) if data else 0)

    def _initialize_positions(self) -> None:
        for symbol in self.symbols:
            self.positions[symbol] = PositionState(symbol=symbol)

    def _seed_market_events(self) -> None:
        """Push the first bar event for each symbol to bootstrap the queue."""
        if self._data_handler is not None:
            for event in self._data_handler.get_next_bars():
                self.event_queue.push(event)

    def _compute_portfolio_equity(self) -> float:
        equity = self.cash
        for symbol, pos in self.positions.items():
            equity += pos.market_value
        return equity

    def run(self) -> Dict[str, Any]:
        """
        Main backtest loop. Processes all events until the queue is empty.
        Returns a results dict with equity curve, trade log, and metadata.
        """
        self._start_time = time.perf_counter()
        self._initialize_positions()
        self._seed_market_events()

        logger.info(
            "Starting backtest: capital=%.2f, symbols=%s",
            self.initial_capital,
            self.symbols,
        )

        while not self.event_queue.empty():
            event = self.event_queue.pop()
            self.process_event(event)

            # After each market event, record equity
            if event.event_type == EventType.MARKET:
                equity = self._compute_portfolio_equity()
                self.equity_curve.record(event.timestamp, equity, self.cash)

        elapsed = time.perf_counter() - (self._start_time or 0)
        logger.info(
            "Backtest complete in %.2fs: %d bars, %.2f total return",
            elapsed,
            self._bar_count,
            self.equity_curve.total_return,
        )

        return self._build_results()

    def process_event(self, event: BaseEvent) -> None:
        """Dispatch an event to all registered handlers."""
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                result = handler(event)
                # If handler returns new events, push them
                if result is not None:
                    if isinstance(result, list):
                        for e in result:
                            if isinstance(e, BaseEvent):
                                self.event_queue.push(e)
                    elif isinstance(result, BaseEvent):
                        self.event_queue.push(result)
            except Exception as exc:
                logger.error(
                    "Handler %s raised on %s: %s",
                    getattr(handler, "__name__", repr(handler)),
                    event.event_type.name,
                    exc,
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Internal default event handlers
    # ------------------------------------------------------------------

    def _on_market(self, event: MarketEvent) -> None:
        """Update position mark-to-market prices and ADV tracker."""
        if not isinstance(event, MarketEvent):
            return
        self._bar_count += 1
        symbol = event.symbol
        if symbol not in self.positions:
            self.positions[symbol] = PositionState(symbol=symbol)
        self.positions[symbol].update_market(event.close)
        self.adv_tracker.update(symbol, event.volume)

        if self.verbose and self._bar_count % 500 == 0:
            logger.debug(
                "Bar %d | %s @ %.4f | equity=%.2f",
                self._bar_count,
                symbol,
                event.close,
                self._compute_portfolio_equity(),
            )

        # If data handler can stream more events, request next bars
        if self._data_handler is not None:
            for next_event in self._data_handler.get_next_bars():
                self.event_queue.push(next_event)

    def _on_fill(self, event: FillEvent) -> None:
        """Apply fill to position state and update cash."""
        if not isinstance(event, FillEvent):
            return
        symbol = event.symbol
        if symbol not in self.positions:
            self.positions[symbol] = PositionState(symbol=symbol)

        pos = self.positions[symbol]
        if pos.entry_time is None and event.quantity != 0:
            pos.entry_time = event.timestamp

        realized = pos.apply_fill(event.quantity, event.fill_price)

        # Cash changes: buying costs cash, selling adds cash
        cash_delta = -event.quantity * event.fill_price - event.commission - event.slippage
        self.cash += cash_delta

        # Log trade
        self.trade_log.append(
            {
                "timestamp": event.timestamp,
                "symbol": symbol,
                "quantity": event.quantity,
                "fill_price": event.fill_price,
                "commission": event.commission,
                "slippage": event.slippage,
                "realized_pnl": realized,
                "cash_after": self.cash,
            }
        )

        logger.debug(
            "FILL %s qty=%.4f @ %.4f | comm=%.4f | slip=%.4f | cash=%.2f",
            symbol,
            event.quantity,
            event.fill_price,
            event.commission,
            event.slippage,
            self.cash,
        )

    # ------------------------------------------------------------------
    # Convenience: generate fill directly from an order (used by execution)
    # ------------------------------------------------------------------

    def generate_fill(
        self,
        order: OrderEvent,
        fill_price: float,
        adv_override: Optional[float] = None,
        daily_vol_override: Optional[float] = None,
    ) -> FillEvent:
        """Create a FillEvent from an OrderEvent, applying cost model."""
        adv = adv_override or self.adv_tracker.get_adv(order.symbol)
        daily_vol = daily_vol_override or 0.02

        comm, slip = self.cost_model.total_cost(
            price=fill_price,
            quantity=order.quantity,
            adv=adv,
            daily_vol=daily_vol,
            is_maker=False,
            is_buy=order.quantity > 0,
        )

        fill = FillEvent(
            event_type=EventType.FILL,
            timestamp=order.timestamp,
            symbol=order.symbol,
            order_id=order.order_id,
            direction=order.direction,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=comm,
            slippage=slip,
        )
        return fill

    # ------------------------------------------------------------------
    # Results assembly
    # ------------------------------------------------------------------

    def _build_results(self) -> Dict[str, Any]:
        ec_df = self.equity_curve.to_dataframe()
        trade_df = pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame()

        results = {
            "equity_curve": ec_df,
            "trade_log": trade_df,
            "total_return": self.equity_curve.total_return,
            "max_drawdown": self.equity_curve.max_drawdown,
            "final_equity": self.equity_curve.current_equity,
            "bars_processed": self._bar_count,
            "num_trades": len(self.trade_log),
            "final_positions": {
                sym: {
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                }
                for sym, pos in self.positions.items()
            },
        }
        return results

    # ------------------------------------------------------------------
    # Utility: run from raw OHLCV dict without a DataHandler
    # ------------------------------------------------------------------

    def run_simple(
        self,
        bars: Dict[str, pd.DataFrame],
        signal_fn: Callable[[MarketEvent], Optional[SignalEvent]],
        sizer_fn: Optional[Callable[[SignalEvent, float], float]] = None,
    ) -> Dict[str, Any]:
        """
        Simplified run mode for single-strategy testing.
        bars: {symbol: DataFrame with columns [open, high, low, close, volume]}
        signal_fn: function(MarketEvent) -> Optional[SignalEvent]
        sizer_fn: function(SignalEvent, cash) -> quantity
        """
        self._initialize_positions()

        # Build chronological sequence of market events
        all_events: List[MarketEvent] = []
        for symbol, df in bars.items():
            if symbol not in self.symbols:
                self.symbols.append(symbol)
                self.positions[symbol] = PositionState(symbol=symbol)
            for ts, row in df.iterrows():
                evt = MarketEvent(
                    event_type=EventType.MARKET,
                    timestamp=pd.Timestamp(ts),
                    symbol=symbol,
                    open=float(row.get("open", row.get("Open", 0))),
                    high=float(row.get("high", row.get("High", 0))),
                    low=float(row.get("low", row.get("Low", 0))),
                    close=float(row.get("close", row.get("Close", 0))),
                    volume=float(row.get("volume", row.get("Volume", 0))),
                )
                all_events.append(evt)

        all_events.sort(key=lambda e: (e.timestamp, e.symbol))

        for mkt_event in all_events:
            self._on_market(mkt_event)
            equity = self._compute_portfolio_equity()
            self.equity_curve.record(mkt_event.timestamp, equity, self.cash)

            sig = signal_fn(mkt_event)
            if sig is None:
                continue

            # Size position
            if sizer_fn is not None:
                qty = sizer_fn(sig, self.cash)
            else:
                # Default: risk 2% of equity per trade
                risk_amount = equity * 0.02
                qty = risk_amount / mkt_event.close if mkt_event.close > 0 else 0.0
                if sig.direction == Direction.SHORT:
                    qty = -qty
                elif sig.direction in (Direction.EXIT, Direction.FLAT):
                    qty = -self.positions[sig.symbol].quantity

            if abs(qty) < 1e-8:
                continue

            # Create and process fill directly (simplified execution)
            order = OrderEvent(
                event_type=EventType.ORDER,
                timestamp=mkt_event.timestamp,
                symbol=sig.symbol,
                order_type=OrderType.MARKET,
                quantity=qty,
                price=mkt_event.close,
                direction=sig.direction,
            )
            fill = self.generate_fill(order, mkt_event.close)
            self._on_fill(fill)

        return self._build_results()

    # ------------------------------------------------------------------
    # Checks and validation
    # ------------------------------------------------------------------

    def check_margin(self, symbol: str) -> bool:
        """Return True if adding the current position doesn't breach max leverage."""
        equity = self._compute_portfolio_equity()
        if equity <= 0:
            return False
        gross_exposure = sum(abs(p.market_value) for p in self.positions.values())
        leverage = gross_exposure / equity
        return leverage <= self.max_leverage

    def get_position(self, symbol: str) -> PositionState:
        if symbol not in self.positions:
            self.positions[symbol] = PositionState(symbol=symbol)
        return self.positions[symbol]

    def get_portfolio_weights(self) -> Dict[str, float]:
        equity = self._compute_portfolio_equity()
        if equity <= 0:
            return {}
        return {
            sym: pos.market_value / equity
            for sym, pos in self.positions.items()
            if abs(pos.market_value) > 0
        }

    def reset(self) -> None:
        """Reset engine state for a fresh backtest run."""
        self.event_queue.clear()
        self.equity_curve = EquityCurve(self.initial_capital)
        self.cash = self.initial_capital
        self.positions = {}
        self.trade_log = []
        self._bar_count = 0
        self._start_time = None
        if self._data_handler is not None:
            self._data_handler.reset()
        logger.info("Engine reset.")
