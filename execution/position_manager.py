"""
execution/position_manager.py
==============================
Production position manager for all open positions.

Tracks every position with full P&L accounting, mark-to-market,
concentration limits, and age-in-bars calculation.  All state can be
persisted to and restored from SQLite so that the trader survives
restarts without losing position data.

Key classes
-----------
Position           -- Value object for a single instrument holding
PositionManager    -- Thread-safe / async-safe collection of positions
PositionPersistence-- SQLite-backed persistence and trade history log

Usage::

    from execution.position_manager import PositionManager, PositionPersistence

    persistence = PositionPersistence(Path("execution/live_trades.db"))
    mgr = PositionManager(persistence=persistence)

    # Restore positions from previous session
    mgr.load_from_db()

    # Open a long position
    await mgr.open_position("SPY", qty=100, fill_price=450.25, commission=0.0)

    # Mark to market
    await mgr.update_prices({"SPY": 451.10})

    # Close partial
    pnl = await mgr.close_position("SPY", qty=50, fill_price=451.10, commission=0.0)
"""

from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger("execution.position_manager")

# ---------------------------------------------------------------------------
# Position dataclass
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """
    Real-time state of a single instrument holding.

    Attributes
    ----------
    symbol : str
        Ticker or coin symbol.
    qty : float
        Signed net quantity (positive = long, negative = short).
    avg_entry_px : float
        Dollar-weighted average entry price of the current holding.
    current_px : float
        Most recent mark-to-market price.
    entry_time : datetime
        UTC timestamp when the position was first opened (or last
        flipped from flat).
    last_update : datetime
        UTC timestamp of the last mark-to-market.
    unrealized_pnl : float
        (current_px - avg_entry_px) * qty
    realized_pnl : float
        Running total of closed P&L for this symbol (session-lifetime).
    cost_basis : float
        abs(qty) * avg_entry_px -- the total dollar cost of the holding.
    commission_paid : float
        Cumulative commissions paid on this position (all fills).
    """

    symbol:          str
    qty:             float   = 0.0
    avg_entry_px:    float   = 0.0
    current_px:      float   = 0.0
    entry_time:      datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_update:     datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    unrealized_pnl:  float   = 0.0
    realized_pnl:    float   = 0.0
    cost_basis:      float   = 0.0
    commission_paid: float   = 0.0

    # ------------------------------------------------------------------ #
    # Derived properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def market_value(self) -> float:
        """Current mark-to-market value (signed)."""
        return self.qty * self.current_px

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    @property
    def is_short(self) -> bool:
        return self.qty < 0

    @property
    def is_flat(self) -> bool:
        return abs(self.qty) < 1e-9

    @property
    def total_pnl(self) -> float:
        """Unrealized + realized P&L, net of commissions."""
        return self.unrealized_pnl + self.realized_pnl - self.commission_paid

    # ------------------------------------------------------------------ #
    # Mark-to-market                                                       #
    # ------------------------------------------------------------------ #

    def mark(self, price: float) -> None:
        """Update current_px and recompute unrealized P&L."""
        self.current_px     = price
        self.unrealized_pnl = (price - self.avg_entry_px) * self.qty
        self.cost_basis     = abs(self.qty) * self.avg_entry_px
        self.last_update    = datetime.now(timezone.utc)

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "symbol":          self.symbol,
            "qty":             self.qty,
            "avg_entry_px":    self.avg_entry_px,
            "current_px":      self.current_px,
            "entry_time":      self.entry_time.isoformat(),
            "last_update":     self.last_update.isoformat(),
            "unrealized_pnl":  round(self.unrealized_pnl,  4),
            "realized_pnl":    round(self.realized_pnl,    4),
            "cost_basis":      round(self.cost_basis,      4),
            "commission_paid": round(self.commission_paid, 4),
            "market_value":    round(self.market_value,    4),
            "total_pnl":       round(self.total_pnl,       4),
        }


# ---------------------------------------------------------------------------
# PositionManager
# ---------------------------------------------------------------------------

class PositionManager:
    """
    Thread-safe and asyncio-compatible manager for all open positions.

    Positions are stored in an internal dict keyed by symbol.
    All mutating methods are async and acquire an asyncio.Lock.
    A threading.RLock is also maintained for synchronous callers that
    cannot use await (e.g. signal handlers, health check threads).

    Parameters
    ----------
    persistence : PositionPersistence | None
        If provided, every open/close is logged to SQLite.  A snapshot
        is written on ``save_snapshot()`` and restored on ``load_from_db()``.
    equity_usd : float
        Reference total equity used for concentration calculations.
        Updated via ``set_equity()``.
    """

    def __init__(
        self,
        persistence: Optional["PositionPersistence"] = None,
        equity_usd:  float = 100_000.0,
    ) -> None:
        self._positions:   dict[str, Position] = {}
        self._lock         = asyncio.Lock()
        self._sync_lock    = threading.RLock()
        self._persistence  = persistence
        self._equity_usd   = equity_usd

    # ------------------------------------------------------------------ #
    # Equity reference                                                     #
    # ------------------------------------------------------------------ #

    def set_equity(self, equity_usd: float) -> None:
        """Update the reference total equity (used for concentration)."""
        with self._sync_lock:
            self._equity_usd = equity_usd

    def get_equity(self) -> float:
        with self._sync_lock:
            return self._equity_usd

    # ------------------------------------------------------------------ #
    # Open / add to position                                               #
    # ------------------------------------------------------------------ #

    async def open_position(
        self,
        symbol:      str,
        qty:         float,
        fill_price:  float,
        commission:  float = 0.0,
        reason:      str   = "",
    ) -> Position:
        """
        Open a new position or add to an existing one.

        Uses dollar-weighted average for entry price when adding.

        Parameters
        ----------
        symbol : str
        qty : float       Signed quantity (positive = buy, negative = sell).
        fill_price : float
        commission : float  Dollar commission for this fill.
        reason : str       Optional free-text reason (logged to SQLite).

        Returns
        -------
        Position
            Updated position state after the fill.
        """
        async with self._lock:
            return self._open_sync(symbol, qty, fill_price, commission, reason)

    def open_position_sync(
        self,
        symbol:     str,
        qty:        float,
        fill_price: float,
        commission: float = 0.0,
        reason:     str   = "",
    ) -> Position:
        """Synchronous version for non-async callers."""
        with self._sync_lock:
            return self._open_sync(symbol, qty, fill_price, commission, reason)

    def _open_sync(
        self,
        symbol:     str,
        qty:        float,
        fill_price: float,
        commission: float,
        reason:     str,
    ) -> Position:
        pos = self._positions.get(symbol)

        if pos is None or pos.is_flat:
            # New position
            pos = Position(
                symbol       = symbol,
                qty          = qty,
                avg_entry_px = fill_price,
                current_px   = fill_price,
                entry_time   = datetime.now(timezone.utc),
            )
            pos.cost_basis     = abs(qty) * fill_price
            pos.commission_paid = commission
        else:
            # Adding to existing position (same side or flip)
            old_cost       = abs(pos.qty) * pos.avg_entry_px
            new_cost       = abs(qty) * fill_price
            new_qty        = pos.qty + qty

            if abs(new_qty) > 1e-9:
                # Weighted average entry
                pos.avg_entry_px = (old_cost + new_cost) / abs(new_qty)
            else:
                pos.avg_entry_px = fill_price

            pos.qty             = new_qty
            pos.commission_paid += commission
            pos.cost_basis       = abs(new_qty) * pos.avg_entry_px

        pos.mark(fill_price)
        self._positions[symbol] = pos

        if self._persistence:
            action = "buy" if qty > 0 else "sell"
            self._persistence.log_trade(
                symbol=symbol, action=action, qty=qty,
                price=fill_price, pnl=0.0, reason=reason,
            )

        log.debug(
            "open_position: %s qty=%.4f @ %.4f avg_entry=%.4f",
            symbol, qty, fill_price, pos.avg_entry_px,
        )
        return pos

    # ------------------------------------------------------------------ #
    # Close / reduce position                                              #
    # ------------------------------------------------------------------ #

    async def close_position(
        self,
        symbol:     str,
        qty:        float,
        fill_price: float,
        commission: float = 0.0,
        reason:     str   = "",
    ) -> float:
        """
        Reduce or close a position and compute realised P&L.

        Parameters
        ----------
        symbol : str
        qty : float
            Absolute quantity to close (sign is inferred from current
            position direction).  If qty > abs(current_qty) the position
            is fully closed (no reversal).
        fill_price : float
        commission : float
        reason : str

        Returns
        -------
        float
            Realised P&L for this closing fill, net of commission.
        """
        async with self._lock:
            return self._close_sync(symbol, qty, fill_price, commission, reason)

    def close_position_sync(
        self,
        symbol:     str,
        qty:        float,
        fill_price: float,
        commission: float = 0.0,
        reason:     str   = "",
    ) -> float:
        """Synchronous version for non-async callers."""
        with self._sync_lock:
            return self._close_sync(symbol, qty, fill_price, commission, reason)

    def _close_sync(
        self,
        symbol:     str,
        qty:        float,
        fill_price: float,
        commission: float,
        reason:     str,
    ) -> float:
        pos = self._positions.get(symbol)
        if pos is None or pos.is_flat:
            log.warning("close_position: no open position for %s", symbol)
            return 0.0

        # qty is the closing quantity (absolute); cap at current holding
        close_qty = min(abs(qty), abs(pos.qty))
        sign      = 1.0 if pos.is_long else -1.0

        # P&L on the closed portion
        pnl = sign * close_qty * (fill_price - pos.avg_entry_px) - commission
        pos.realized_pnl    += pnl
        pos.commission_paid += commission

        # Reduce position
        pos.qty -= sign * close_qty
        if abs(pos.qty) < 1e-9:
            pos.qty = 0.0

        pos.cost_basis = abs(pos.qty) * pos.avg_entry_px
        pos.mark(fill_price)
        self._positions[symbol] = pos

        if self._persistence:
            action = "close_long" if sign > 0 else "close_short"
            self._persistence.log_trade(
                symbol=symbol, action=action, qty=close_qty,
                price=fill_price, pnl=pnl, reason=reason,
            )

        log.debug(
            "close_position: %s qty=%.4f @ %.4f realised_pnl=%.4f remaining=%.4f",
            symbol, close_qty, fill_price, pnl, pos.qty,
        )
        return pnl

    # ------------------------------------------------------------------ #
    # Mark-to-market                                                       #
    # ------------------------------------------------------------------ #

    async def update_prices(self, prices: dict[str, float]) -> None:
        """
        Mark all positions to market using a price dict.

        Parameters
        ----------
        prices : dict[str, float]
            Map of symbol -> latest price.  Symbols not in the dict are
            left unchanged.
        """
        async with self._lock:
            self._update_prices_sync(prices)

    def update_prices_sync(self, prices: dict[str, float]) -> None:
        """Synchronous mark-to-market."""
        with self._sync_lock:
            self._update_prices_sync(prices)

    def _update_prices_sync(self, prices: dict[str, float]) -> None:
        for symbol, price in prices.items():
            pos = self._positions.get(symbol)
            if pos is not None:
                pos.mark(price)

    # ------------------------------------------------------------------ #
    # Queries                                                              #
    # ------------------------------------------------------------------ #

    def get_position(self, symbol: str) -> Optional[Position]:
        """Return the Position for *symbol*, or None if flat."""
        pos = self._positions.get(symbol)
        return pos if (pos and not pos.is_flat) else None

    def all_positions(self) -> list[Position]:
        """Return a list of all non-flat positions."""
        return [p for p in self._positions.values() if not p.is_flat]

    async def get_portfolio_summary(self) -> dict:
        """
        Return a summary of the current portfolio.

        Returns
        -------
        dict with keys:
            "total_market_value" : float
            "unrealized_pnl"     : float
            "realized_pnl"       : float
            "total_pnl"          : float
            "n_positions"        : int
            "positions"          : list[dict]
            "long_exposure"      : float
            "short_exposure"     : float
        """
        async with self._lock:
            return self._portfolio_summary_sync()

    def get_portfolio_summary_sync(self) -> dict:
        with self._sync_lock:
            return self._portfolio_summary_sync()

    def _portfolio_summary_sync(self) -> dict:
        positions = [p for p in self._positions.values() if not p.is_flat]
        total_mv  = sum(p.market_value    for p in positions)
        unreal    = sum(p.unrealized_pnl  for p in positions)
        realized  = sum(p.realized_pnl    for p in positions)
        commiss   = sum(p.commission_paid for p in positions)
        long_exp  = sum(p.market_value for p in positions if p.is_long)
        short_exp = sum(abs(p.market_value) for p in positions if p.is_short)

        return {
            "total_market_value": round(total_mv,  4),
            "unrealized_pnl":     round(unreal,    4),
            "realized_pnl":       round(realized,  4),
            "total_pnl":          round(unreal + realized - commiss, 4),
            "commission_paid":    round(commiss,   4),
            "n_positions":        len(positions),
            "long_exposure":      round(long_exp,  4),
            "short_exposure":     round(short_exp, 4),
            "positions":          [p.to_dict() for p in positions],
        }

    def get_position_age_bars(
        self,
        symbol:            str,
        bar_duration_mins: float = 15.0,
    ) -> int:
        """
        Return number of bars elapsed since position entry.

        Parameters
        ----------
        bar_duration_mins : float
            Duration of each bar in minutes.  Default 15 min.

        Returns
        -------
        int
            Number of complete bars since entry_time.
            Returns 0 if no position or entry_time is not set.
        """
        pos = self._positions.get(symbol)
        if pos is None or pos.is_flat:
            return 0
        now      = datetime.now(timezone.utc)
        elapsed  = (now - pos.entry_time).total_seconds() / 60.0
        return int(elapsed / bar_duration_mins)

    def get_concentration(self, symbol: str) -> float:
        """
        Return the position's market value as a fraction of total equity.

        Parameters
        ----------
        symbol : str

        Returns
        -------
        float
            Concentration in [0, 1].  0.0 if no position or equity == 0.
        """
        pos = self._positions.get(symbol)
        if pos is None or pos.is_flat:
            return 0.0
        equity = self.get_equity()
        if equity <= 0:
            return 0.0
        return abs(pos.market_value) / equity

    def export_to_dict(self) -> list[dict]:
        """Return a list of position dicts for serialisation to SQLite."""
        return [p.to_dict() for p in self._positions.values() if not p.is_flat]

    # ------------------------------------------------------------------ #
    # Persistence helpers                                                  #
    # ------------------------------------------------------------------ #

    def save_snapshot(self) -> None:
        """Persist current positions to SQLite via PositionPersistence."""
        if self._persistence is None:
            return
        self._persistence.save_snapshot(self._positions)

    def load_from_db(self) -> int:
        """
        Restore positions from the last SQLite snapshot.

        Returns the number of positions loaded.
        """
        if self._persistence is None:
            return 0
        loaded = self._persistence.load_snapshot()
        for symbol, pos in loaded.items():
            self._positions[symbol] = pos
        log.info("PositionManager: loaded %d positions from DB", len(loaded))
        return len(loaded)


# ---------------------------------------------------------------------------
# PositionPersistence
# ---------------------------------------------------------------------------

_SCHEMA_POSITIONS = """
CREATE TABLE IF NOT EXISTS position_snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL    NOT NULL,
    symbol       TEXT    NOT NULL,
    qty          REAL    NOT NULL,
    avg_entry_px REAL    NOT NULL,
    current_px   REAL    NOT NULL,
    entry_time   TEXT    NOT NULL,
    unrealized   REAL    NOT NULL,
    realized     REAL    NOT NULL,
    cost_basis   REAL    NOT NULL,
    commission   REAL    NOT NULL
)
"""

_SCHEMA_TRADES = """
CREATE TABLE IF NOT EXISTS trade_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL    NOT NULL,
    symbol     TEXT    NOT NULL,
    action     TEXT    NOT NULL,
    qty        REAL    NOT NULL,
    price      REAL    NOT NULL,
    pnl        REAL    NOT NULL,
    reason     TEXT    NOT NULL
)
"""

_IDX_POS_TS  = "CREATE INDEX IF NOT EXISTS idx_ps_ts     ON position_snapshots (ts)"
_IDX_POS_SYM = "CREATE INDEX IF NOT EXISTS idx_ps_symbol ON position_snapshots (symbol)"
_IDX_TRD_TS  = "CREATE INDEX IF NOT EXISTS idx_tl_ts     ON trade_log (ts)"
_IDX_TRD_SYM = "CREATE INDEX IF NOT EXISTS idx_tl_symbol ON trade_log (symbol)"


class PositionPersistence:
    """
    SQLite-backed store for position snapshots and trade history.

    The snapshot table stores the latest known state per symbol so that
    the system can recover after a restart without re-processing all fills.

    Trade log entries are append-only and capture every fill with symbol,
    action, qty, price, P&L, and optional reason string.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Snapshot                                                             #
    # ------------------------------------------------------------------ #

    def save_snapshot(self, positions: dict[str, Position]) -> None:
        """
        Persist all non-flat positions.

        Overwrites the existing snapshot for each symbol (upsert via
        DELETE + INSERT for SQLite compatibility).
        """
        non_flat = {s: p for s, p in positions.items() if not p.is_flat}
        if not non_flat:
            return

        with self._lock:
            con = sqlite3.connect(str(self._db_path))
            try:
                ts_now = time.time()
                for symbol, pos in non_flat.items():
                    # Delete existing snapshot for this symbol
                    con.execute(
                        "DELETE FROM position_snapshots WHERE symbol = ?", (symbol,)
                    )
                    con.execute(
                        """
                        INSERT INTO position_snapshots
                            (ts, symbol, qty, avg_entry_px, current_px, entry_time,
                             unrealized, realized, cost_basis, commission)
                        VALUES (?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            ts_now,
                            symbol,
                            pos.qty,
                            pos.avg_entry_px,
                            pos.current_px,
                            pos.entry_time.isoformat(),
                            pos.unrealized_pnl,
                            pos.realized_pnl,
                            pos.cost_basis,
                            pos.commission_paid,
                        ),
                    )
                con.commit()
            finally:
                con.close()
        log.info("PositionPersistence: saved snapshot for %d symbols", len(non_flat))

    def load_snapshot(self) -> dict[str, Position]:
        """
        Load the most recent snapshot for each symbol.

        Returns
        -------
        dict[str, Position]
            Positions keyed by symbol.
        """
        with self._lock:
            con = sqlite3.connect(str(self._db_path))
            try:
                rows = con.execute(
                    """
                    SELECT symbol, qty, avg_entry_px, current_px, entry_time,
                           unrealized, realized, cost_basis, commission
                    FROM position_snapshots
                    ORDER BY ts DESC
                    """
                ).fetchall()
            finally:
                con.close()

        # De-duplicate: keep latest row per symbol (rows are ts-ordered desc)
        seen: set[str] = set()
        positions: dict[str, Position] = {}
        for row in rows:
            symbol = row[0]
            if symbol in seen:
                continue
            seen.add(symbol)

            try:
                entry_time = datetime.fromisoformat(row[4])
            except ValueError:
                entry_time = datetime.now(timezone.utc)

            pos = Position(
                symbol          = symbol,
                qty             = row[1],
                avg_entry_px    = row[2],
                current_px      = row[3],
                entry_time      = entry_time,
                unrealized_pnl  = row[5],
                realized_pnl    = row[6],
                cost_basis      = row[7],
                commission_paid = row[8],
            )
            pos.last_update = datetime.now(timezone.utc)
            positions[symbol] = pos

        log.info("PositionPersistence: loaded %d position(s) from snapshot", len(positions))
        return positions

    def delete_snapshot(self, symbol: str) -> None:
        """Remove snapshot rows for *symbol* (call after position is fully closed)."""
        with self._lock:
            con = sqlite3.connect(str(self._db_path))
            con.execute("DELETE FROM position_snapshots WHERE symbol = ?", (symbol,))
            con.commit()
            con.close()

    # ------------------------------------------------------------------ #
    # Trade log                                                            #
    # ------------------------------------------------------------------ #

    def log_trade(
        self,
        symbol: str,
        action: str,
        qty:    float,
        price:  float,
        pnl:    float,
        reason: str = "",
    ) -> None:
        """
        Append a trade record to the trade log.

        Parameters
        ----------
        symbol : str
        action : str     e.g. "buy", "sell", "close_long", "close_short"
        qty : float      Absolute quantity.
        price : float    Fill price.
        pnl : float      Realised P&L for this fill (0 for opening fills).
        reason : str     Optional strategy signal or reason string.
        """
        with self._lock:
            try:
                con = sqlite3.connect(str(self._db_path))
                con.execute(
                    "INSERT INTO trade_log (ts, symbol, action, qty, price, pnl, reason) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (time.time(), symbol, action, qty, price, pnl, reason),
                )
                con.commit()
                con.close()
            except Exception as exc:
                log.error("PositionPersistence.log_trade failed: %s", exc)

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        since:  Optional[float] = None,
    ) -> list[dict]:
        """
        Retrieve trade history.

        Parameters
        ----------
        symbol : str | None     Filter to this symbol.
        since : float | None    Filter to records after this Unix timestamp.

        Returns
        -------
        list[dict]
            Each dict has keys: ts, symbol, action, qty, price, pnl, reason.
        """
        conditions = []
        params: list = []
        if symbol is not None:
            conditions.append("symbol = ?")
            params.append(symbol)
        if since is not None:
            conditions.append("ts >= ?")
            params.append(since)

        where  = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        query  = f"SELECT ts, symbol, action, qty, price, pnl, reason FROM trade_log {where} ORDER BY ts"

        with self._lock:
            con  = sqlite3.connect(str(self._db_path))
            rows = con.execute(query, params).fetchall()
            con.close()

        return [
            {
                "ts":     row[0],
                "symbol": row[1],
                "action": row[2],
                "qty":    row[3],
                "price":  row[4],
                "pnl":    row[5],
                "reason": row[6],
            }
            for row in rows
        ]

    def get_cumulative_pnl(self, symbol: Optional[str] = None) -> float:
        """Return total realised P&L from the trade log."""
        conditions = []
        params: list = []
        if symbol is not None:
            conditions.append("symbol = ?")
            params.append(symbol)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        query = f"SELECT COALESCE(SUM(pnl), 0.0) FROM trade_log {where}"
        with self._lock:
            con = sqlite3.connect(str(self._db_path))
            result = con.execute(query, params).fetchone()
            con.close()
        return result[0] if result else 0.0

    def get_trade_count(self, symbol: Optional[str] = None) -> int:
        """Return total number of trade log entries."""
        conditions = []
        params: list = []
        if symbol is not None:
            conditions.append("symbol = ?")
            params.append(symbol)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        query = f"SELECT COUNT(*) FROM trade_log {where}"
        with self._lock:
            con = sqlite3.connect(str(self._db_path))
            result = con.execute(query, params).fetchone()
            con.close()
        return result[0] if result else 0

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        con = sqlite3.connect(str(self._db_path))
        con.execute(_SCHEMA_POSITIONS)
        con.execute(_SCHEMA_TRADES)
        con.execute(_IDX_POS_TS)
        con.execute(_IDX_POS_SYM)
        con.execute(_IDX_TRD_TS)
        con.execute(_IDX_TRD_SYM)
        con.commit()
        con.close()
