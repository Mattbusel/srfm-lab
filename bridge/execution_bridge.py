"""
bridge/execution_bridge.py
===========================
Bridge between the live trader's execution decisions and the cost model /
smart router infrastructure.

Architecture
------------
  ExecutionBridge
    - Intercepts order intents from the live trader before submission
    - Estimates execution cost via execution/cost_model.py CostEstimator
    - Routes via execution/smart_router.py SmartRouter for optimal venue/type
    - Records routing decision, estimated cost, and actual fill outcome
    - Tracks rolling 30-day implementation shortfall (IS) per symbol
    - Exposes read-only execution quality metrics for observability

  ExecutionQualityTracker
    - Maintains per-symbol rolling deques of (estimated_bps, actual_bps) pairs
    - Computes rolling 30-day average IS = actual_bps - estimated_bps
    - Persists IS records to SQLite for long-term analysis

Usage::

    from bridge.execution_bridge import ExecutionBridge

    bridge = ExecutionBridge()

    # Intercept before sending to Alpaca
    decisions = bridge.on_order_intent(symbol="SPY", qty=100.0, urgency=0.5)
    # ... execute decisions via broker ...
    # Record actual fill
    bridge.on_fill(decisions[0], actual_fill={"fill_price": 450.12, "fill_qty": 100.0})
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Deque, Optional

log = logging.getLogger("bridge.execution_bridge")

_REPO_ROOT = Path(__file__).parents[1]
_EXEC_DB = _REPO_ROOT / "execution" / "live_trades.db"
_BRIDGE_DB = _REPO_ROOT / "execution" / "execution_bridge.db"

# IS rolling window: 30 calendar days
_IS_WINDOW_DAYS = 30
_IS_DEQUE_SIZE = 500   # max records per symbol in memory


# ---------------------------------------------------------------------------
# Try to import cost model and smart router
# ---------------------------------------------------------------------------

try:
    import sys as _sys
    _sys.path.insert(0, str(_REPO_ROOT))
    from execution.cost_model import CostEstimator, CostEstimate, VENUES
    from execution.smart_router import SmartRouter, OrderIntent, RoutingDecision, LiquidityMap
    _EXEC_AVAILABLE = True
    log.debug("ExecutionBridge: loaded cost_model and smart_router")
except ImportError as _e:
    log.warning("ExecutionBridge: execution modules not available: %s -- using stubs", _e)
    _EXEC_AVAILABLE = False
    CostEstimator = None  # type: ignore[assignment,misc]
    CostEstimate = None   # type: ignore[assignment,misc]
    SmartRouter = None    # type: ignore[assignment,misc]
    OrderIntent = None    # type: ignore[assignment,misc]
    RoutingDecision = None  # type: ignore[assignment,misc]
    LiquidityMap = None   # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BridgeOrderIntent:
    """
    Order intent as received from the live trader before routing.

    Parameters
    ----------
    symbol : str
        Instrument ticker (e.g., "SPY", "BTC/USD").
    qty : float
        Signed quantity: positive = buy, negative = sell.
    urgency : float
        0.0 = maximally patient, 1.0 = immediate.
    asset_class : str
        "equity" or "crypto".
    adv_usd : float
        Estimated 30-day average daily volume USD (used for impact model).
    sigma_daily : float
        Daily volatility fraction.
    ref_price : float
        Reference price at time of intent (for IS computation).
    strategy_id : str
        Strategy identifier for attribution.
    signal_id : str
        Signal that generated this intent.
    """

    symbol: str
    qty: float
    urgency: float = 0.5
    asset_class: str = "equity"
    adv_usd: float = 1_000_000.0
    sigma_daily: float = 0.02
    ref_price: float = 0.0
    strategy_id: str = "larsa_v18"
    signal_id: str = ""
    intent_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "symbol":      self.symbol,
            "qty":         self.qty,
            "urgency":     self.urgency,
            "asset_class": self.asset_class,
            "adv_usd":     self.adv_usd,
            "sigma_daily": self.sigma_daily,
            "ref_price":   self.ref_price,
            "strategy_id": self.strategy_id,
            "signal_id":   self.signal_id,
            "intent_time": self.intent_time,
        }


@dataclass
class BridgeRoutingDecision:
    """
    Result of routing an order intent through the execution bridge.

    Wraps the SmartRouter RoutingDecision with bridge-specific metadata.
    """

    intent: BridgeOrderIntent
    venue: str
    order_type: str
    limit_price: float
    qty: float
    bar_index: int
    estimated_cost_bps: float
    reasoning: str
    cost_estimate: dict = field(default_factory=dict)  # full CostEstimate dict
    decision_id: str = ""
    decided_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "decision_id":         self.decision_id,
            "symbol":              self.intent.symbol,
            "qty":                 self.qty,
            "venue":               self.venue,
            "order_type":          self.order_type,
            "limit_price":         self.limit_price,
            "bar_index":           self.bar_index,
            "estimated_cost_bps":  round(self.estimated_cost_bps, 4),
            "reasoning":           self.reasoning,
            "decided_at":          self.decided_at,
            "cost_breakdown":      self.cost_estimate,
            "intent":              self.intent.to_dict(),
        }


@dataclass
class FillRecord:
    """Actual fill data received after order execution."""

    decision_id: str
    symbol: str
    fill_price: float
    fill_qty: float
    fill_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    venue: str = ""
    commission_paid: float = 0.0
    slippage_bps: float = 0.0  # computed: (fill_price - ref_price) / ref_price * 10000

    def to_dict(self) -> dict:
        return {
            "decision_id":   self.decision_id,
            "symbol":        self.symbol,
            "fill_price":    self.fill_price,
            "fill_qty":      self.fill_qty,
            "fill_time":     self.fill_time,
            "venue":         self.venue,
            "commission":    self.commission_paid,
            "slippage_bps":  round(self.slippage_bps, 4),
        }


@dataclass
class ISRecord:
    """Implementation shortfall record: estimated vs actual cost per fill."""

    symbol: str
    decision_id: str
    estimated_bps: float
    actual_bps: float
    is_bps: float   # implementation shortfall = actual - estimated
    ref_price: float
    fill_price: float
    qty: float
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "symbol":        self.symbol,
            "decision_id":   self.decision_id,
            "estimated_bps": round(self.estimated_bps, 4),
            "actual_bps":    round(self.actual_bps, 4),
            "is_bps":        round(self.is_bps, 4),
            "ref_price":     self.ref_price,
            "fill_price":    self.fill_price,
            "qty":           self.qty,
            "recorded_at":   self.recorded_at,
        }


# ---------------------------------------------------------------------------
# Stub classes (used when execution modules not available)
# ---------------------------------------------------------------------------

class _StubCostEstimate:
    total_bps: float = 2.0
    commission_bps: float = 0.0
    spread_cost_bps: float = 1.0
    impact_bps: float = 1.0
    timing_bps: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_bps": self.total_bps,
            "commission_bps": self.commission_bps,
            "spread_cost_bps": self.spread_cost_bps,
            "impact_bps": self.impact_bps,
        }


class _StubRoutingDecision:
    venue: str = "alpaca_equity"
    order_type: str = "limit"
    limit_price: float = 0.0
    qty: float = 0.0
    bar_index: int = 0
    estimated_cost_bps: float = 2.0
    reasoning: str = "stub routing (execution modules unavailable)"


# ---------------------------------------------------------------------------
# ExecutionQualityTracker
# ---------------------------------------------------------------------------

_QUALITY_DDL = """
CREATE TABLE IF NOT EXISTS implementation_shortfall (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT    NOT NULL,
    decision_id  TEXT    NOT NULL,
    estimated_bps REAL   NOT NULL,
    actual_bps   REAL    NOT NULL,
    is_bps       REAL    NOT NULL,
    ref_price    REAL    NOT NULL,
    fill_price   REAL    NOT NULL,
    qty          REAL    NOT NULL,
    recorded_at  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_is_symbol ON implementation_shortfall (symbol);
CREATE INDEX IF NOT EXISTS idx_is_date ON implementation_shortfall (recorded_at);
"""


class ExecutionQualityTracker:
    """
    Tracks per-symbol implementation shortfall (IS) with a rolling 30-day window.

    IS = actual_cost_bps - estimated_cost_bps

    Positive IS means we paid more than expected (execution was worse than
    the pre-trade estimate).  Negative IS means favourable execution.
    """

    def __init__(self, db_path: Path = _BRIDGE_DB) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # In-memory rolling deques: symbol -> deque of IS values (bps)
        self._is_deques: dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=_IS_DEQUE_SIZE)
        )
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.executescript(_QUALITY_DDL)

    def record(self, rec: ISRecord) -> None:
        """Persist an IS record and update the in-memory rolling window."""
        with self._lock:
            self._is_deques[rec.symbol].append(rec.is_bps)
            try:
                with self._conn() as conn:
                    conn.execute(
                        """
                        INSERT INTO implementation_shortfall
                          (symbol, decision_id, estimated_bps, actual_bps, is_bps,
                           ref_price, fill_price, qty, recorded_at)
                        VALUES (?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            rec.symbol,
                            rec.decision_id,
                            rec.estimated_bps,
                            rec.actual_bps,
                            rec.is_bps,
                            rec.ref_price,
                            rec.fill_price,
                            rec.qty,
                            rec.recorded_at,
                        ),
                    )
            except Exception as exc:
                log.warning("ExecutionQualityTracker: DB write failed: %s", exc)

    def get_rolling_is(self, symbol: str) -> dict:
        """
        Return rolling IS statistics for a symbol.

        Returns dict with: mean_is_bps, std_is_bps, count, min_is_bps, max_is_bps.
        """
        with self._lock:
            vals = list(self._is_deques.get(symbol, []))
        if not vals:
            return {"mean_is_bps": 0.0, "std_is_bps": 0.0, "count": 0,
                    "min_is_bps": 0.0, "max_is_bps": 0.0}
        n = len(vals)
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
        return {
            "symbol":      symbol,
            "mean_is_bps": round(mean, 4),
            "std_is_bps":  round(math.sqrt(variance), 4),
            "count":       n,
            "min_is_bps":  round(min(vals), 4),
            "max_is_bps":  round(max(vals), 4),
        }

    def get_30d_is_from_db(self, symbol: str) -> dict:
        """Query the DB for the last 30 days of IS for a symbol."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=_IS_WINDOW_DAYS)).isoformat()
        with self._lock:
            try:
                with self._conn() as conn:
                    rows = conn.execute(
                        """
                        SELECT is_bps FROM implementation_shortfall
                        WHERE symbol = ? AND recorded_at > ?
                        ORDER BY recorded_at ASC
                        """,
                        (symbol, cutoff),
                    ).fetchall()
            except Exception as exc:
                log.warning("ExecutionQualityTracker: DB query failed: %s", exc)
                return {"symbol": symbol, "count": 0}

        vals = [float(r["is_bps"]) for r in rows]
        if not vals:
            return {"symbol": symbol, "count": 0, "mean_is_bps": 0.0}
        n = len(vals)
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
        return {
            "symbol":      symbol,
            "count":       n,
            "mean_is_bps": round(mean, 4),
            "std_is_bps":  round(math.sqrt(variance), 4),
            "min_is_bps":  round(min(vals), 4),
            "max_is_bps":  round(max(vals), 4),
            "window_days": _IS_WINDOW_DAYS,
        }

    def get_all_symbols(self) -> list[str]:
        """Return all symbols with IS records in memory."""
        with self._lock:
            return list(self._is_deques.keys())


# ---------------------------------------------------------------------------
# ExecutionBridge
# ---------------------------------------------------------------------------

class ExecutionBridge:
    """
    Intercepts live trader order intents and routes them through the cost
    model and smart router before broker submission.

    Thread safety
    -------------
    on_order_intent and on_fill are synchronous and thread-safe.
    They can be called from any thread (live trader uses threads).
    Internal state is protected by threading.Lock.

    Decision ID format
    ------------------
    {symbol}_{timestamp_ms} -- deterministic, no external dependencies.
    """

    def __init__(
        self,
        db_path: Path = _BRIDGE_DB,
        exec_db_path: Path = _EXEC_DB,
    ) -> None:
        self._quality_tracker = ExecutionQualityTracker(db_path)
        self._exec_db_path = exec_db_path
        self._lock = threading.Lock()

        # Pending decisions awaiting fill: decision_id -> BridgeRoutingDecision
        self._pending: dict[str, BridgeRoutingDecision] = {}

        # Instantiate cost model and smart router if available
        if _EXEC_AVAILABLE:
            self._cost_estimator = CostEstimator()
            self._smart_router = SmartRouter()
            self._liquidity_map = LiquidityMap()
        else:
            self._cost_estimator = None
            self._smart_router = None
            self._liquidity_map = None

        log.info(
            "ExecutionBridge: initialised (exec_modules=%s)",
            "available" if _EXEC_AVAILABLE else "stub",
        )

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def on_order_intent(
        self,
        symbol: str,
        qty: float,
        urgency: float = 0.5,
        asset_class: str = "equity",
        adv_usd: float = 1_000_000.0,
        sigma_daily: float = 0.02,
        ref_price: float = 0.0,
        strategy_id: str = "larsa_v18",
        signal_id: str = "",
    ) -> list[BridgeRoutingDecision]:
        """
        Process an order intent from the live trader.

        Steps
        -----
        1. Build BridgeOrderIntent
        2. Estimate pre-trade cost via CostEstimator
        3. Route via SmartRouter
        4. Record decision in pending dict
        5. Return list of BridgeRoutingDecision (one per child order)

        Returns an empty list on critical failure.
        """
        intent = BridgeOrderIntent(
            symbol=symbol,
            qty=qty,
            urgency=urgency,
            asset_class=asset_class,
            adv_usd=adv_usd,
            sigma_daily=sigma_daily,
            ref_price=ref_price,
            strategy_id=strategy_id,
            signal_id=signal_id,
        )

        try:
            decisions = self._route_intent(intent)
        except Exception as exc:
            log.error("ExecutionBridge: routing failed for %s: %s", symbol, exc)
            decisions = [self._emergency_fallback_decision(intent)]

        with self._lock:
            for d in decisions:
                self._pending[d.decision_id] = d

        return decisions

    def on_fill(
        self,
        routing_decision: BridgeRoutingDecision,
        actual_fill: dict[str, Any],
    ) -> FillRecord | None:
        """
        Record the actual fill outcome against a routing decision.

        Computes actual slippage in bps and updates the IS tracker.

        Parameters
        ----------
        routing_decision : BridgeRoutingDecision
        actual_fill : dict with keys fill_price, fill_qty, [commission_paid], [venue]

        Returns FillRecord with computed slippage, or None on error.
        """
        try:
            return self._process_fill(routing_decision, actual_fill)
        except Exception as exc:
            log.error("ExecutionBridge: fill processing failed: %s", exc)
            return None

    def update_liquidity(
        self,
        symbol: str,
        venue: str,
        bid: float,
        ask: float,
        bid_size: float = 0.0,
        ask_size: float = 0.0,
    ) -> None:
        """Feed live quote data to the liquidity map for the smart router."""
        if self._liquidity_map is not None:
            try:
                self._liquidity_map.update_quote(
                    symbol, venue, bid=bid, ask=ask,
                    bid_size=bid_size, ask_size=ask_size,
                )
            except Exception as exc:
                log.debug("ExecutionBridge: liquidity update failed: %s", exc)

    def get_execution_quality(self, symbol: str) -> dict:
        """Return rolling IS statistics for a symbol."""
        return self._quality_tracker.get_rolling_is(symbol)

    def get_30d_quality(self, symbol: str) -> dict:
        """Return 30-day IS statistics from the DB for a symbol."""
        return self._quality_tracker.get_30d_is_from_db(symbol)

    def get_all_quality_report(self) -> list[dict]:
        """Return quality metrics for all tracked symbols."""
        return [self._quality_tracker.get_rolling_is(sym)
                for sym in self._quality_tracker.get_all_symbols()]

    def get_pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    # ------------------------------------------------------------------
    # Internal routing
    # ------------------------------------------------------------------

    def _route_intent(self, intent: BridgeOrderIntent) -> list[BridgeRoutingDecision]:
        """
        Route an intent through cost model + smart router.
        Returns list of child BridgeRoutingDecisions.
        """
        cost_est = self._estimate_cost(intent)
        raw_decisions = self._smart_route(intent, cost_est)
        return raw_decisions

    def _estimate_cost(self, intent: BridgeOrderIntent) -> Any:
        """Run pre-trade cost estimation. Returns CostEstimate or stub."""
        if self._cost_estimator is None:
            return _StubCostEstimate()
        try:
            side = "buy" if intent.qty > 0 else "sell"
            price = intent.ref_price if intent.ref_price > 0 else 100.0
            order_size_usd = abs(intent.qty) * price
            venue = "alpaca_crypto" if intent.asset_class == "crypto" else "alpaca_equity"
            est = self._cost_estimator.estimate(
                symbol=intent.symbol,
                order_size_usd=order_size_usd,
                side=side,
                venue=venue,
                adv_usd=intent.adv_usd,
                sigma_daily=intent.sigma_daily,
            )
            log.debug(
                "ExecutionBridge: cost estimate %s -- total=%.2fbps (spread=%.2f impact=%.2f)",
                intent.symbol,
                est.total_bps,
                est.spread_cost_bps,
                est.impact_bps,
            )
            return est
        except Exception as exc:
            log.warning("ExecutionBridge: cost estimation failed: %s", exc)
            return _StubCostEstimate()

    def _smart_route(
        self, intent: BridgeOrderIntent, cost_est: Any
    ) -> list[BridgeRoutingDecision]:
        """
        Route via SmartRouter. Falls back to a single limit order if unavailable.
        """
        if self._smart_router is None or OrderIntent is None:
            return [self._stub_routing_decision(intent, cost_est)]

        try:
            raw_intent = OrderIntent(
                symbol=intent.symbol,
                target_qty=intent.qty,
                urgency=intent.urgency,
                adv_usd=intent.adv_usd,
                sigma_daily=intent.sigma_daily,
                asset_class=intent.asset_class,
                max_slippage_bps=max(5.0, cost_est.total_bps * 1.5),
            )
            if intent.ref_price > 0:
                raw_intent.price_limit = _compute_price_limit(
                    intent.ref_price, intent.qty, slippage_frac=0.005
                )

            raw_decisions = self._smart_router.route(raw_intent, self._liquidity_map)
            bridge_decisions = []
            for rd in (raw_decisions if isinstance(raw_decisions, list) else [raw_decisions]):
                bridge_decisions.append(self._wrap_routing_decision(intent, rd, cost_est))
            return bridge_decisions if bridge_decisions else [self._stub_routing_decision(intent, cost_est)]

        except Exception as exc:
            log.warning("ExecutionBridge: SmartRouter failed: %s", exc)
            return [self._stub_routing_decision(intent, cost_est)]

    def _wrap_routing_decision(
        self,
        intent: BridgeOrderIntent,
        raw_rd: Any,
        cost_est: Any,
    ) -> BridgeRoutingDecision:
        decision_id = _make_decision_id(intent.symbol)
        return BridgeRoutingDecision(
            intent=intent,
            venue=str(raw_rd.venue),
            order_type=str(raw_rd.order_type),
            limit_price=float(raw_rd.limit_price),
            qty=float(raw_rd.qty),
            bar_index=int(raw_rd.bar_index),
            estimated_cost_bps=float(raw_rd.estimated_cost_bps),
            reasoning=str(getattr(raw_rd, "reasoning", "")),
            cost_estimate=cost_est.to_dict() if hasattr(cost_est, "to_dict") else {},
            decision_id=decision_id,
        )

    def _stub_routing_decision(
        self,
        intent: BridgeOrderIntent,
        cost_est: Any,
    ) -> BridgeRoutingDecision:
        """Fallback routing when SmartRouter unavailable."""
        venue = "alpaca_crypto" if intent.asset_class == "crypto" else "alpaca_equity"
        order_type = "market" if intent.urgency >= 0.8 else "limit"
        limit_price = 0.0
        if order_type == "limit" and intent.ref_price > 0:
            slippage = 0.001  # 10 bps
            if intent.qty > 0:
                limit_price = intent.ref_price * (1 + slippage)
            else:
                limit_price = intent.ref_price * (1 - slippage)

        reasoning = (
            f"stub_routing: venue={venue} type={order_type} "
            f"urgency={intent.urgency:.2f} estimated_cost={cost_est.total_bps:.2f}bps"
        )
        return BridgeRoutingDecision(
            intent=intent,
            venue=venue,
            order_type=order_type,
            limit_price=limit_price,
            qty=intent.qty,
            bar_index=0,
            estimated_cost_bps=float(cost_est.total_bps),
            reasoning=reasoning,
            cost_estimate=cost_est.to_dict() if hasattr(cost_est, "to_dict") else {},
            decision_id=_make_decision_id(intent.symbol),
        )

    def _emergency_fallback_decision(self, intent: BridgeOrderIntent) -> BridgeRoutingDecision:
        """Last-resort decision when everything else fails."""
        venue = "alpaca_crypto" if intent.asset_class == "crypto" else "alpaca_equity"
        return BridgeRoutingDecision(
            intent=intent,
            venue=venue,
            order_type="market",
            limit_price=0.0,
            qty=intent.qty,
            bar_index=0,
            estimated_cost_bps=5.0,
            reasoning="emergency_fallback: all routing systems failed",
            cost_estimate={},
            decision_id=_make_decision_id(intent.symbol),
        )

    # ------------------------------------------------------------------
    # Fill processing
    # ------------------------------------------------------------------

    def _process_fill(
        self,
        routing_decision: BridgeRoutingDecision,
        actual_fill: dict[str, Any],
    ) -> FillRecord:
        fill_price = float(actual_fill.get("fill_price", actual_fill.get("price", 0.0)))
        fill_qty = float(actual_fill.get("fill_qty", actual_fill.get("qty", routing_decision.qty)))
        commission = float(actual_fill.get("commission_paid", actual_fill.get("commission", 0.0)))
        venue = str(actual_fill.get("venue", routing_decision.venue))

        # Compute actual slippage in bps
        ref_price = routing_decision.intent.ref_price
        if ref_price > 0 and fill_price > 0:
            price_diff = fill_price - ref_price
            # For sells, positive slip = bad
            if routing_decision.qty < 0:
                price_diff = -price_diff
            slippage_bps = (price_diff / ref_price) * 10000.0
        else:
            slippage_bps = 0.0

        # Commission in bps (if ref_price known)
        notional = abs(fill_qty) * fill_price if fill_price > 0 else 0.0
        commission_bps = (commission / notional * 10000.0) if notional > 0 else 0.0

        actual_cost_bps = slippage_bps + commission_bps
        estimated_bps = routing_decision.estimated_cost_bps
        is_bps = actual_cost_bps - estimated_bps

        fill_rec = FillRecord(
            decision_id=routing_decision.decision_id,
            symbol=routing_decision.intent.symbol,
            fill_price=fill_price,
            fill_qty=fill_qty,
            venue=venue,
            commission_paid=commission,
            slippage_bps=round(slippage_bps, 4),
        )

        is_rec = ISRecord(
            symbol=routing_decision.intent.symbol,
            decision_id=routing_decision.decision_id,
            estimated_bps=round(estimated_bps, 4),
            actual_bps=round(actual_cost_bps, 4),
            is_bps=round(is_bps, 4),
            ref_price=ref_price,
            fill_price=fill_price,
            qty=fill_qty,
        )

        self._quality_tracker.record(is_rec)

        # Remove from pending
        with self._lock:
            self._pending.pop(routing_decision.decision_id, None)

        log.info(
            "ExecutionBridge: fill %s %s -- est=%.2fbps actual=%.2fbps IS=%.2fbps",
            routing_decision.intent.symbol,
            routing_decision.decision_id,
            estimated_bps,
            actual_cost_bps,
            is_bps,
        )

        if abs(is_bps) > 20.0:
            log.warning(
                "ExecutionBridge: large IS for %s -- IS=%.2fbps (est=%.2f actual=%.2f)",
                routing_decision.intent.symbol,
                is_bps,
                estimated_bps,
                actual_cost_bps,
            )

        return fill_rec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_decision_id(symbol: str) -> str:
    ts_ms = int(time.time() * 1000)
    clean = symbol.replace("/", "_")
    return f"{clean}_{ts_ms}"


def _compute_price_limit(ref_price: float, qty: float, slippage_frac: float) -> float:
    """Return a conservative price limit for a limit order."""
    if qty > 0:
        return ref_price * (1.0 + slippage_frac)
    return ref_price * (1.0 - slippage_frac)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s UTC [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    bridge = ExecutionBridge()

    # Demo: simulate a round trip
    decisions = bridge.on_order_intent(
        symbol="SPY",
        qty=100.0,
        urgency=0.5,
        asset_class="equity",
        ref_price=450.0,
        adv_usd=25_000_000.0,
        sigma_daily=0.012,
    )
    log.info("Decisions: %s", [d.to_dict() for d in decisions])

    if decisions:
        fill = bridge.on_fill(decisions[0], {"fill_price": 450.05, "fill_qty": 100.0})
        if fill:
            log.info("Fill: %s", fill.to_dict())

    quality = bridge.get_execution_quality("SPY")
    log.info("Quality: %s", quality)


if __name__ == "__main__":
    _main()
