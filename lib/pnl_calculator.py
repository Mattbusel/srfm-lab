"""
pnl_calculator.py -- real-time P&L calculation engine.

Handles FIFO cost basis, multi-source attribution (symbol, signal, regime),
and daily session aggregation. Produces JSON-serializable PnLReport.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Core data types
# ------------------------------------------------------------------

@dataclass
class Trade:
    """Completed round-trip trade record."""
    trade_id: str
    symbol: str
    side: str              # BUY (long entry) | SELL (short entry)
    qty: float
    entry_price: float
    exit_price: float
    entry_ts: int
    exit_ts: int
    signal_source: Optional[str] = None   # which signal generated this trade
    regime: Optional[str] = None          # regime at time of entry
    realized_pnl: float = 0.0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FillRecord:
    """Individual fill for FIFO cost-basis tracking."""
    qty: float
    price: float
    ts: int


# ------------------------------------------------------------------
# FIFO cost basis tracker
# ------------------------------------------------------------------

class FIFOCostBasis:
    """
    Maintains a FIFO queue of fill records to compute average cost basis
    and realized P&L on partial or full exits.
    """

    def __init__(self):
        self._queue: deque[FillRecord] = deque()
        self._total_qty: float = 0.0

    def add_fill(self, qty: float, price: float, ts: int = 0) -> None:
        """Add an entry fill to the FIFO queue."""
        if qty <= 0:
            raise ValueError(f"fill qty must be positive, got {qty}")
        self._queue.append(FillRecord(qty=qty, price=price, ts=ts))
        self._total_qty += qty

    def compute_cost_basis(self, qty: float) -> float:
        """
        Return the weighted average cost for `qty` units using FIFO ordering.
        Raises ValueError if qty > available inventory.
        """
        if qty > self._total_qty + 1e-9:
            raise ValueError(
                f"requested qty {qty} > available {self._total_qty}"
            )
        remaining = qty
        total_cost = 0.0
        for fill in self._queue:
            if remaining <= 0:
                break
            consumed = min(remaining, fill.qty)
            total_cost += consumed * fill.price
            remaining -= consumed
        return total_cost / qty if qty > 0 else 0.0

    def consume(self, qty: float) -> float:
        """
        Remove qty units from the FIFO queue (on exit).
        Returns the weighted average cost of consumed units.
        """
        if qty > self._total_qty + 1e-9:
            raise ValueError(
                f"consume qty {qty} > available {self._total_qty}"
            )
        remaining = qty
        total_cost = 0.0
        while self._queue and remaining > 1e-9:
            fill = self._queue[0]
            consumed = min(remaining, fill.qty)
            total_cost += consumed * fill.price
            fill.qty -= consumed
            remaining -= consumed
            self._total_qty -= consumed
            if fill.qty < 1e-9:
                self._queue.popleft()
        avg_cost = total_cost / qty if qty > 0 else 0.0
        return avg_cost

    @property
    def total_qty(self) -> float:
        return self._total_qty

    @property
    def average_cost(self) -> float:
        if self._total_qty < 1e-9:
            return 0.0
        total_cost = sum(f.qty * f.price for f in self._queue)
        return total_cost / self._total_qty


# ------------------------------------------------------------------
# Per-symbol P&L state
# ------------------------------------------------------------------

class SymbolPnLState:
    """Tracks open position and running P&L for a single symbol."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.long_basis  = FIFOCostBasis()
        self.short_basis = FIFOCostBasis()
        self.net_qty     = 0.0       # positive = long, negative = short
        self.realized    = 0.0
        self.signal_source: Optional[str] = None
        self.regime: Optional[str] = None

    def open_long(self, qty: float, price: float, ts: int) -> None:
        self.long_basis.add_fill(qty, price, ts)
        self.net_qty += qty

    def open_short(self, qty: float, price: float, ts: int) -> None:
        self.short_basis.add_fill(qty, price, ts)
        self.net_qty -= qty

    def close_long(self, qty: float, price: float, ts: int) -> float:
        """Close qty of a long position. Returns realized P&L."""
        avg_cost = self.long_basis.consume(qty)
        pnl = (price - avg_cost) * qty
        self.net_qty -= qty
        self.realized += pnl
        return pnl

    def close_short(self, qty: float, price: float, ts: int) -> float:
        """Close qty of a short position. Returns realized P&L."""
        avg_cost = self.short_basis.consume(qty)
        pnl = (avg_cost - price) * qty
        self.net_qty += qty
        self.realized += pnl
        return pnl

    def unrealized_pnl(self, current_price: float) -> float:
        """Compute mark-to-market unrealized P&L at current_price."""
        upnl = 0.0
        if self.long_basis.total_qty > 1e-9:
            upnl += (current_price - self.long_basis.average_cost) * self.long_basis.total_qty
        if self.short_basis.total_qty > 1e-9:
            upnl += (self.short_basis.average_cost - current_price) * self.short_basis.total_qty
        return upnl


# ------------------------------------------------------------------
# P&L calculator
# ------------------------------------------------------------------

class PnLCalculator:
    """
    Computes unrealized and realized P&L in real time.
    Handles multi-currency, corporate actions, and dividends.
    """

    def __init__(self):
        self._symbols: dict[str, SymbolPnLState] = {}
        self._completed_trades: list[Trade] = []
        self._fx_rates: dict[str, float] = {}   # currency -> USD conversion rate
        self._session_start_ts: int = int(time.time())

    def _get_or_create(self, symbol: str) -> SymbolPnLState:
        if symbol not in self._symbols:
            self._symbols[symbol] = SymbolPnLState(symbol)
        return self._symbols[symbol]

    # ------------------------------------------------------------------
    # Entry / exit recording
    # ------------------------------------------------------------------

    def record_entry(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        ts: int,
        signal_source: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> None:
        """Record an entry fill (long or short)."""
        state = self._get_or_create(symbol)
        state.signal_source = signal_source
        state.regime = regime
        if side.upper() == "BUY":
            state.open_long(qty, price, ts)
        else:
            state.open_short(qty, price, ts)

    def record_exit(
        self,
        symbol: str,
        side: str,       # BUY = closing a short, SELL = closing a long
        qty: float,
        price: float,
        ts: int,
        trade_id: Optional[str] = None,
    ) -> float:
        """
        Record an exit fill. Returns realized P&L for this fill.
        """
        state = self._get_or_create(symbol)
        if side.upper() == "SELL":
            rpnl = state.close_long(qty, price, ts)
        else:
            rpnl = state.close_short(qty, price, ts)

        # -- record completed trade
        trade = Trade(
            trade_id=trade_id or f"{symbol}_{ts}",
            symbol=symbol,
            side="LONG" if side.upper() == "SELL" else "SHORT",
            qty=qty,
            entry_price=0.0,   # -- not tracked here; see FIFOCostBasis
            exit_price=price,
            entry_ts=0,
            exit_ts=ts,
            signal_source=state.signal_source,
            regime=state.regime,
            realized_pnl=rpnl,
        )
        self._completed_trades.append(trade)
        return rpnl

    # ------------------------------------------------------------------
    # Static P&L helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_realized_pnl(trade: Trade) -> float:
        """
        Recompute realized P&L from a Trade record.
        Uses entry/exit prices (independent of FIFO state).
        """
        if trade.side == "LONG":
            return (trade.exit_price - trade.entry_price) * trade.qty
        else:
            return (trade.entry_price - trade.exit_price) * trade.qty

    @staticmethod
    def compute_unrealized_pnl(
        symbol: str, qty: float, avg_cost: float, current_price: float
    ) -> float:
        """
        Simple unrealized P&L given position parameters.
        qty > 0 = long, qty < 0 = short.
        """
        if qty > 0:
            return (current_price - avg_cost) * qty
        elif qty < 0:
            return (avg_cost - current_price) * abs(qty)
        return 0.0

    # ------------------------------------------------------------------
    # Aggregate queries
    # ------------------------------------------------------------------

    def total_unrealized_pnl(self, prices: dict[str, float]) -> float:
        """Sum unrealized P&L across all open positions."""
        total = 0.0
        for sym, state in self._symbols.items():
            price = prices.get(sym)
            if price is None:
                continue
            total += state.unrealized_pnl(price)
        return total

    def total_realized_pnl(self) -> float:
        return sum(s.realized for s in self._symbols.values())

    def apply_corporate_action(
        self, symbol: str, split_ratio: float = 1.0, dividend: float = 0.0
    ) -> None:
        """
        Adjust FIFO queue for splits (adjust price by 1/ratio, qty by ratio)
        or dividends (add cash to realized P&L).
        """
        state = self._symbols.get(symbol)
        if state is None:
            return
        if dividend > 0.0 and state.net_qty > 0:
            state.realized += dividend * state.long_basis.total_qty
            logger.info("dividend %.4f applied to %s", dividend, symbol)
        if split_ratio != 1.0 and split_ratio > 0:
            # -- adjust all FIFO fills: qty *= ratio, price /= ratio
            for basis in (state.long_basis, state.short_basis):
                for fill in basis._queue:
                    fill.qty   *= split_ratio
                    fill.price /= split_ratio
                basis._total_qty *= split_ratio
            logger.info("split %.2f applied to %s", split_ratio, symbol)

    def set_fx_rate(self, currency: str, usd_rate: float) -> None:
        """Register an FX conversion rate (e.g. EUR -> 1.08 means 1 EUR = 1.08 USD)."""
        self._fx_rates[currency] = usd_rate

    def convert_to_usd(self, amount: float, currency: str) -> float:
        """Convert amount in currency to USD using registered FX rate."""
        rate = self._fx_rates.get(currency, 1.0)
        return amount * rate


# ------------------------------------------------------------------
# Daily P&L aggregator
# ------------------------------------------------------------------

class DailyPnLAggregator:
    """
    Resets at session open; accumulates intraday P&L entries
    by symbol, signal source, and regime.
    """

    def __init__(self):
        self._entries: list[dict] = []
        self._session_ts: int = int(time.time())

    def reset(self) -> None:
        self._entries.clear()
        self._session_ts = int(time.time())

    def record(
        self,
        symbol: str,
        pnl: float,
        pnl_type: str,          # realized | unrealized
        signal_source: Optional[str] = None,
        regime: Optional[str] = None,
        ts: Optional[int] = None,
    ) -> None:
        self._entries.append({
            "symbol":        symbol,
            "pnl":           pnl,
            "pnl_type":      pnl_type,
            "signal_source": signal_source or "unknown",
            "regime":        regime or "unknown",
            "ts":            ts or int(time.time()),
        })

    def total(self) -> float:
        return sum(e["pnl"] for e in self._entries)

    def by_symbol(self) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for e in self._entries:
            out[e["symbol"]] += e["pnl"]
        return dict(out)

    def by_signal_source(self) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for e in self._entries:
            out[e["signal_source"]] += e["pnl"]
        return dict(out)

    def by_regime(self) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for e in self._entries:
            out[e["regime"]] += e["pnl"]
        return dict(out)

    def by_type(self) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for e in self._entries:
            out[e["pnl_type"]] += e["pnl"]
        return dict(out)


# ------------------------------------------------------------------
# P&L attribution
# ------------------------------------------------------------------

class PnLAttribution:
    """
    Slice P&L across multiple dimensions for risk and performance review.
    """

    def __init__(self, aggregator: DailyPnLAggregator):
        self._agg = aggregator

    def by_symbol(self) -> dict[str, float]:
        return self._agg.by_symbol()

    def by_signal_source(self) -> dict[str, float]:
        return self._agg.by_signal_source()

    def by_regime(self) -> dict[str, float]:
        return self._agg.by_regime()

    def best_symbol(self) -> Optional[str]:
        sym_pnl = self.by_symbol()
        if not sym_pnl:
            return None
        return max(sym_pnl, key=lambda s: sym_pnl[s])

    def worst_symbol(self) -> Optional[str]:
        sym_pnl = self.by_symbol()
        if not sym_pnl:
            return None
        return min(sym_pnl, key=lambda s: sym_pnl[s])


# ------------------------------------------------------------------
# P&L report
# ------------------------------------------------------------------

@dataclass
class PnLReport:
    """JSON report with all P&L attribution slices."""
    session_start_ts: int
    report_ts: int
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    by_symbol: dict[str, float]
    by_signal_source: dict[str, float]
    by_regime: dict[str, float]
    trade_count: int
    best_symbol: Optional[str]
    worst_symbol: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def build_pnl_report(
    calculator: PnLCalculator,
    aggregator: DailyPnLAggregator,
    prices: dict[str, float],
) -> PnLReport:
    """
    Build a PnLReport from a PnLCalculator + DailyPnLAggregator snapshot.
    """
    attribution = PnLAttribution(aggregator)
    realized    = calculator.total_realized_pnl()
    unrealized  = calculator.total_unrealized_pnl(prices)

    return PnLReport(
        session_start_ts=aggregator._session_ts,
        report_ts=int(time.time()),
        total_pnl=realized + unrealized,
        realized_pnl=realized,
        unrealized_pnl=unrealized,
        by_symbol=attribution.by_symbol(),
        by_signal_source=attribution.by_signal_source(),
        by_regime=attribution.by_regime(),
        trade_count=len(calculator._completed_trades),
        best_symbol=attribution.best_symbol(),
        worst_symbol=attribution.worst_symbol(),
    )
