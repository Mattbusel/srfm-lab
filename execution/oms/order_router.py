"""
execution/oms/order_router.py
==============================
Order routing logic within the SRFM OMS layer.

The OrderRouter sits between the OrderManager and the broker adapters.  It:
  1. Runs pre-trade checks via the risk_management interface.
  2. Selects the correct broker (Alpaca for equities, Binance for crypto,
     simulated for futures).
  3. Applies notional and ADV-based size caps.
  4. Splits oversized orders into TWAP child slices.
  5. Logs every routing decision with full rationale.

Routing rules
-------------
  Asset class  | Broker     | Max notional per order | Split threshold
  -------------|------------|------------------------|----------------
  equity       | alpaca     | $50,000                | 10 % of ADV
  crypto       | binance    | $25,000                | 5 % of ADV
  futures      | simulated  | $100,000               | 20 % of ADV

ADV defaults
------------
  If no ADV is provided, the system uses $5M (equity) / $50M (crypto) as
  conservative defaults so size caps are computed even without market data.

TWAP slicing
------------
  When an order exceeds the split threshold, it is broken into N slices
  where N = ceil(qty / max_slice_qty).  Each slice is assigned a
  delay_seconds offset for the TWAP executor.

All routing decisions are logged to RoutingAuditLog which writes to SQLite.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("execution.order_router")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

-- max notional per single child order (equity / crypto / futures)
MAX_NOTIONAL_EQUITY:  float = 50_000.0
MAX_NOTIONAL_CRYPTO:  float = 25_000.0
MAX_NOTIONAL_FUTURES: float = 100_000.0

-- default ADV assumptions when no market data is available
DEFAULT_ADV_EQUITY:  float = 5_000_000.0
DEFAULT_ADV_CRYPTO:  float = 50_000_000.0

-- fraction of ADV above which order is split into TWAP slices
ADV_SPLIT_THRESHOLD_EQUITY:  float = 0.10   -- 10 %
ADV_SPLIT_THRESHOLD_CRYPTO:  float = 0.05   -- 5 %

-- default TWAP slice inter-arrival gap in seconds
DEFAULT_SLICE_DELAY_SECONDS: float = 30.0

ROUTING_DB_PATH = Path(__file__).parent.parent / "routing_audit.db"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass
class OrderSlice:
    """
    A single TWAP child order slice.

    Attributes
    ----------
    slice_id       : Sequential index (0-based)
    qty            : Quantity for this slice
    delay_seconds  : Seconds after T=0 to submit this slice
    parent_order_id: OMS ID of the parent order
    """
    slice_id:        int
    qty:             float
    delay_seconds:   float
    parent_order_id: str

    def to_dict(self) -> dict:
        return {
            "slice_id":        self.slice_id,
            "qty":             self.qty,
            "delay_seconds":   self.delay_seconds,
            "parent_order_id": self.parent_order_id,
        }


@dataclass
class RoutingDecision:
    """
    Result of OrderRouter.route().

    Attributes
    ----------
    order_id           : OMS order ID
    broker             : Target broker name ('alpaca', 'binance', 'simulated')
    execution_strategy : 'DIRECT' or 'TWAP'
    slices             : List of OrderSlice objects (len >= 1)
                         Single-element list for DIRECT orders.
    reason             : Human-readable routing rationale
    pre_trade_passed   : True if all pre-trade risk checks passed
    rejected_reason    : Non-empty string if pre_trade_passed is False
    notional_usd       : Estimated order notional
    adv_fraction       : Fraction of ADV this order represents
    """
    order_id:           str
    broker:             str
    execution_strategy: str
    slices:             List[OrderSlice]
    reason:             str
    pre_trade_passed:   bool  = True
    rejected_reason:    str   = ""
    notional_usd:       float = 0.0
    adv_fraction:       float = 0.0

    def to_dict(self) -> dict:
        return {
            "order_id":           self.order_id,
            "broker":             self.broker,
            "execution_strategy": self.execution_strategy,
            "slices":             [s.to_dict() for s in self.slices],
            "reason":             self.reason,
            "pre_trade_passed":   self.pre_trade_passed,
            "rejected_reason":    self.rejected_reason,
            "notional_usd":       self.notional_usd,
            "adv_fraction":       self.adv_fraction,
        }


# ---------------------------------------------------------------------------
# BrokerSelector
# ---------------------------------------------------------------------------

class BrokerSelector:
    """
    Rules-based broker selection.

    Selection logic
    ---------------
    - Symbol contains '/' (e.g. BTC/USD, ETH/USD) -> crypto -> binance
    - Symbol ends with a digit (futures code e.g. ES2406) -> futures -> simulated
    - Everything else -> equity -> alpaca

    The asset_class hint in the order dict overrides auto-detection if provided.
    """

    CRYPTO_BROKERS:  List[str] = ["binance"]
    EQUITY_BROKERS:  List[str] = ["alpaca"]
    FUTURES_BROKERS: List[str] = ["simulated"]

    @staticmethod
    def detect_asset_class(symbol: str, hint: Optional[str] = None) -> str:
        """Return 'equity', 'crypto', or 'futures'."""
        if hint and hint.lower() in ("equity", "crypto", "futures"):
            return hint.lower()
        if "/" in symbol:
            return "crypto"
        if symbol and symbol[-1].isdigit():
            return "futures"
        return "equity"

    def select(self, order: dict) -> Tuple[str, str]:
        """
        Return (broker_name, asset_class) for the given order dict.

        Reads 'symbol' and optional 'asset_class' from the dict.
        """
        symbol      = order.get("symbol", "")
        asset_hint  = order.get("asset_class")
        asset_class = self.detect_asset_class(symbol, asset_hint)

        if asset_class == "crypto":
            return "binance", "crypto"
        elif asset_class == "futures":
            return "simulated", "futures"
        else:
            return "alpaca", "equity"


# ---------------------------------------------------------------------------
# OrderSplitter
# ---------------------------------------------------------------------------

class OrderSplitter:
    """
    Splits a large order into N child TWAP slices.

    Parameters
    ----------
    slice_delay_seconds : float
        Time gap between consecutive slices (default 30 s).

    The splitter computes N = ceil(total_qty / max_slice_qty) and
    distributes quantity evenly, with the remainder on the last slice.
    """

    def __init__(self, slice_delay_seconds: float = DEFAULT_SLICE_DELAY_SECONDS) -> None:
        self._slice_delay = slice_delay_seconds

    def split(
        self,
        order_id: str,
        total_qty: float,
        max_slice_qty: float,
        delay_seconds: Optional[float] = None,
    ) -> List[OrderSlice]:
        """
        Produce a list of OrderSlice objects.

        Parameters
        ----------
        order_id      : Parent OMS order ID.
        total_qty     : Total order quantity to split.
        max_slice_qty : Maximum quantity per slice.
        delay_seconds : Override for inter-slice delay (uses constructor default if None).

        Returns
        -------
        List of OrderSlice -- at least one element.
        """
        if max_slice_qty <= 0:
            max_slice_qty = total_qty

        delay = delay_seconds if delay_seconds is not None else self._slice_delay
        n_slices  = max(1, math.ceil(total_qty / max_slice_qty))
        base_qty  = total_qty / n_slices
        slices    = []

        for i in range(n_slices):
            -- last slice absorbs rounding remainder
            if i == n_slices - 1:
                qty = total_qty - sum(s.qty for s in slices)
            else:
                qty = base_qty

            slices.append(OrderSlice(
                slice_id        = i,
                qty             = round(max(qty, 0.0), 8),
                delay_seconds   = i * delay,
                parent_order_id = order_id,
            ))

        return slices


# ---------------------------------------------------------------------------
# RoutingAuditLog
# ---------------------------------------------------------------------------

class RoutingAuditLog:
    """
    Persists every routing decision to SQLite for compliance and debugging.

    Schema
    ------
        routing_decisions (
            id          INTEGER PRIMARY KEY,
            decision_ts TEXT,
            order_id    TEXT,
            symbol      TEXT,
            broker      TEXT,
            strategy    TEXT,
            n_slices    INTEGER,
            notional    REAL,
            adv_frac    REAL,
            passed      INTEGER,
            reason      TEXT,
            rejected    TEXT
        )
    """

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self._db_path = Path(db_path) if db_path else ROUTING_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS routing_decisions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_ts TEXT    NOT NULL,
                order_id    TEXT    NOT NULL,
                symbol      TEXT,
                broker      TEXT    NOT NULL,
                strategy    TEXT    NOT NULL,
                n_slices    INTEGER NOT NULL DEFAULT 1,
                notional    REAL    NOT NULL DEFAULT 0.0,
                adv_frac    REAL    NOT NULL DEFAULT 0.0,
                passed      INTEGER NOT NULL DEFAULT 1,
                reason      TEXT,
                rejected    TEXT
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rd_order ON routing_decisions (order_id)"
        )
        self._conn.commit()

    def log(self, order: dict, decision: RoutingDecision) -> None:
        """Persist a RoutingDecision."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO routing_decisions
                    (decision_ts, order_id, symbol, broker, strategy,
                     n_slices, notional, adv_frac, passed, reason, rejected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    decision.order_id,
                    order.get("symbol"),
                    decision.broker,
                    decision.execution_strategy,
                    len(decision.slices),
                    decision.notional_usd,
                    decision.adv_fraction,
                    1 if decision.pre_trade_passed else 0,
                    decision.reason,
                    decision.rejected_reason or None,
                ),
            )
            self._conn.commit()

    def get_recent(self, limit: int = 50) -> List[dict]:
        """Return the most recent routing decisions."""
        cur = self._conn.execute(
            """
            SELECT decision_ts, order_id, symbol, broker, strategy,
                   n_slices, notional, adv_frac, passed, reason, rejected
            FROM routing_decisions
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        )
        cols = ["decision_ts", "order_id", "symbol", "broker", "strategy",
                "n_slices", "notional", "adv_frac", "passed", "reason", "rejected"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# OrderRouter
# ---------------------------------------------------------------------------

class OrderRouter:
    """
    Routes validated orders to appropriate broker adapters.

    Applies pre-trade checks, size constraints, and circuit breaker gates.
    Large orders are automatically split into TWAP child slices.

    Parameters
    ----------
    pre_trade_checker : object | None
        Object with a run_checks(order_dict) -> (bool, str) interface.
        If None, pre-trade checks are skipped (paper trading mode).
    adv_provider : callable | None
        Callable(symbol) -> float that returns the average daily volume
        in shares/units for the given symbol.  If None, defaults are used.
    broker_adapters : dict | None
        Optional {broker_name: adapter_object} map.  When provided,
        route() will call adapter.submit(order_dict) and return the
        broker_order_id.  If None, route() returns a simulated ID.
    db_path : Path | str | None
        Override for the routing audit log SQLite path.
    circuit_breaker : callable | None
        Callable() -> bool that returns True if trading is halted.
    """

    def __init__(
        self,
        pre_trade_checker=None,
        adv_provider=None,
        broker_adapters: Optional[Dict[str, object]] = None,
        db_path: Optional[Path | str] = None,
        circuit_breaker=None,
    ) -> None:
        self._pre_trade    = pre_trade_checker
        self._adv_provider = adv_provider
        self._adapters     = broker_adapters or {}
        self._circuit_breaker = circuit_breaker
        self._selector     = BrokerSelector()
        self._splitter     = OrderSplitter()
        self._audit        = RoutingAuditLog(db_path)
        self._lock         = threading.RLock()

    # ------------------------------------------------------------------
    # ADV helpers
    # ------------------------------------------------------------------

    def _get_adv(self, symbol: str, asset_class: str) -> float:
        """Return ADV for symbol, falling back to asset-class default."""
        if self._adv_provider:
            try:
                adv = self._adv_provider(symbol)
                if adv and adv > 0:
                    return adv
            except Exception as exc:
                log.warning("adv_provider error for %s: %s", symbol, exc)
        return DEFAULT_ADV_CRYPTO if asset_class == "crypto" else DEFAULT_ADV_EQUITY

    # ------------------------------------------------------------------
    # Size cap computation
    # ------------------------------------------------------------------

    def _max_notional(self, asset_class: str) -> float:
        if asset_class == "crypto":
            return MAX_NOTIONAL_CRYPTO
        elif asset_class == "futures":
            return MAX_NOTIONAL_FUTURES
        return MAX_NOTIONAL_EQUITY

    def _adv_split_threshold(self, asset_class: str) -> float:
        if asset_class == "crypto":
            return ADV_SPLIT_THRESHOLD_CRYPTO
        return ADV_SPLIT_THRESHOLD_EQUITY

    def _compute_max_slice_qty(
        self,
        total_qty: float,
        price: float,
        asset_class: str,
        adv_units: float,
    ) -> float:
        """
        Return the maximum quantity per slice based on:
          1. Notional cap: max_notional / price
          2. ADV cap: adv_split_threshold * adv_units

        Returns the smaller of the two constraints.
        """
        max_notional   = self._max_notional(asset_class)
        adv_threshold  = self._adv_split_threshold(asset_class)

        notional_cap = max_notional / price if price > 0 else total_qty
        adv_cap      = adv_units * adv_threshold

        return min(notional_cap, adv_cap, total_qty)

    # ------------------------------------------------------------------
    # Primary routing method
    # ------------------------------------------------------------------

    def route(self, order: dict) -> RoutingDecision:
        """
        Route an order to the appropriate broker with pre-trade checks.

        Parameters
        ----------
        order : dict
            Order dict with at minimum: order_id, symbol, side, qty, price.
            Optional: asset_class, strategy_id.

        Returns
        -------
        RoutingDecision with full routing metadata.
        """
        with self._lock:
            order_id = order.get("order_id", "unknown")
            symbol   = order.get("symbol", "")
            qty      = float(order.get("qty", order.get("quantity", 0.0)))
            price    = float(order.get("price", 0.0))
            if price <= 0:
                price = float(order.get("curr_price", 1.0))

            -- circuit breaker gate
            if self._circuit_breaker:
                try:
                    halted = self._circuit_breaker()
                except Exception:
                    halted = False
                if halted:
                    decision = RoutingDecision(
                        order_id           = order_id,
                        broker             = "none",
                        execution_strategy = "BLOCKED",
                        slices             = [],
                        reason             = "circuit_breaker_tripped",
                        pre_trade_passed   = False,
                        rejected_reason    = "Circuit breaker is active -- trading halted",
                        notional_usd       = qty * price,
                        adv_fraction       = 0.0,
                    )
                    self._audit.log(order, decision)
                    return decision

            -- broker selection
            broker, asset_class = self._selector.select(order)

            -- pre-trade checks
            if self._pre_trade:
                try:
                    passed, reject_reason = self._pre_trade.run_checks(order)
                except Exception as exc:
                    log.error("pre_trade error for %s: %s", symbol, exc)
                    passed, reject_reason = True, ""  -- fail-open on checker error
            else:
                passed, reject_reason = True, ""

            if not passed:
                decision = RoutingDecision(
                    order_id           = order_id,
                    broker             = broker,
                    execution_strategy = "REJECTED",
                    slices             = [],
                    reason             = "pre_trade_check_failed",
                    pre_trade_passed   = False,
                    rejected_reason    = reject_reason,
                    notional_usd       = qty * price,
                    adv_fraction       = 0.0,
                )
                self._audit.log(order, decision)
                log.warning(
                    "OrderRouter rejected order=%s symbol=%s: %s",
                    order_id, symbol, reject_reason,
                )
                return decision

            -- ADV and size cap computation
            adv_units   = self._get_adv(symbol, asset_class)
            notional    = qty * price
            adv_fraction = (qty / adv_units) if adv_units > 0 else 0.0

            max_slice_qty = self._compute_max_slice_qty(
                qty, price, asset_class, adv_units
            )

            -- determine execution strategy
            if qty <= max_slice_qty + 1e-9:
                strategy = "DIRECT"
                slices   = [OrderSlice(
                    slice_id        = 0,
                    qty             = qty,
                    delay_seconds   = 0.0,
                    parent_order_id = order_id,
                )]
                reason = (
                    f"direct route to {broker} -- "
                    f"qty={qty:.4f} notional=${notional:,.0f} "
                    f"adv_frac={adv_fraction:.2%}"
                )
            else:
                strategy = "TWAP"
                slices   = self._splitter.split(
                    order_id      = order_id,
                    total_qty     = qty,
                    max_slice_qty = max_slice_qty,
                )
                reason = (
                    f"TWAP split to {broker} -- "
                    f"qty={qty:.4f} notional=${notional:,.0f} "
                    f"adv_frac={adv_fraction:.2%} "
                    f"n_slices={len(slices)} "
                    f"max_slice_qty={max_slice_qty:.4f}"
                )
                log.info(
                    "OrderRouter TWAP split: order=%s symbol=%s "
                    "n=%d each=%.4f total=%.4f",
                    order_id, symbol, len(slices), max_slice_qty, qty,
                )

            decision = RoutingDecision(
                order_id           = order_id,
                broker             = broker,
                execution_strategy = strategy,
                slices             = slices,
                reason             = reason,
                pre_trade_passed   = True,
                rejected_reason    = "",
                notional_usd       = notional,
                adv_fraction       = adv_fraction,
            )

            self._audit.log(order, decision)

            log.info(
                "OrderRouter: order=%s %s %s -> %s [%s] "
                "notional=$%.0f adv_frac=%.2%%",
                order_id, order.get("side", "?"), symbol,
                broker, strategy, notional, adv_fraction * 100,
            )

            return decision

    # ------------------------------------------------------------------
    # Direct broker dispatch (optional)
    # ------------------------------------------------------------------

    def dispatch(self, order: dict, decision: RoutingDecision) -> List[str]:
        """
        Dispatch order slices to the broker adapter.

        If broker adapters are configured, this method calls
        adapter.submit(slice_order_dict) for each slice and returns
        the list of broker-assigned order IDs.

        Parameters
        ----------
        order    : Original order dict.
        decision : RoutingDecision from route().

        Returns
        -------
        List of broker_order_id strings (one per slice).
        """
        if not decision.pre_trade_passed:
            log.warning(
                "dispatch skipped -- order=%s was rejected", decision.order_id
            )
            return []

        adapter = self._adapters.get(decision.broker)
        broker_ids = []

        for slc in decision.slices:
            slice_order = dict(order)
            slice_order["qty"]             = slc.qty
            slice_order["quantity"]        = slc.qty
            slice_order["parent_order_id"] = slc.parent_order_id
            slice_order["slice_id"]        = slc.slice_id
            slice_order["delay_seconds"]   = slc.delay_seconds

            if adapter:
                try:
                    broker_id = adapter.submit(slice_order)
                    broker_ids.append(broker_id)
                    log.debug(
                        "dispatch: slice %d -> %s broker_id=%s",
                        slc.slice_id, decision.broker, broker_id,
                    )
                except Exception as exc:
                    log.error(
                        "dispatch error: order=%s slice=%d broker=%s: %s",
                        decision.order_id, slc.slice_id, decision.broker, exc,
                    )
                    broker_ids.append(f"ERROR_{slc.slice_id}")
            else:
                -- simulated: generate a fake broker ID
                broker_ids.append(
                    f"SIM_{decision.broker.upper()}_{decision.order_id[:8]}_{slc.slice_id}"
                )

        return broker_ids

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def audit_log(self) -> RoutingAuditLog:
        return self._audit

    def close(self) -> None:
        self._audit.close()
