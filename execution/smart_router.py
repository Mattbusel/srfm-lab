"""
execution/smart_router.py
=========================
Smart order router for equities and crypto.

Selects the optimal venue and execution strategy based on:
  - Order urgency (0 = patient, 1 = immediate)
  - Order size relative to ADV
  - Live bid-ask spread and depth
  - Venue circuit-breaker status
  - Pre-trade cost estimates

Key classes
-----------
OrderIntent         -- What the strategy wants executed
RoutingDecision     -- The router's execution plan for a single child order
LiquidityMap        -- Live quote state per symbol/venue
SmartRouter         -- Core routing logic
DarkPoolChecker     -- Dark pool fill probability estimation
ExecutionLogger     -- SQLite log of all decisions and outcomes

Usage::

    from execution.smart_router import SmartRouter, OrderIntent, LiquidityMap

    liq = LiquidityMap()
    liq.update_quote("SPY", "alpaca_equity", bid=450.10, ask=450.12,
                     bid_size=1000, ask_size=800)

    router = SmartRouter()
    intent = OrderIntent(symbol="SPY", target_qty=500, urgency=0.5,
                         max_slippage_bps=5.0, time_limit_bars=4)
    decisions = router.route(intent, liq)
"""

from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from execution.cost_model import (
    CostEstimator,
    VENUES,
    CostEstimate,
    _almgren_chriss_trajectory,
    _intraday_volume_profile,
)

log = logging.getLogger("execution.smart_router")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OrderIntent:
    """
    Strategy-level order intent.

    Attributes
    ----------
    symbol : str
        Instrument to trade.
    target_qty : float
        Signed target quantity (positive = buy, negative = sell).
    urgency : float
        0.0 = maximally patient (TWAP over full time_limit_bars),
        1.0 = immediate execution (market order).
    max_slippage_bps : float
        Maximum acceptable total cost in basis points.  Router will warn
        but still execute if exceeded (does not hard-cancel).
    time_limit_bars : int
        Number of 15-min bars available to complete the order.
    price_limit : float | None
        Hard price limit for limit orders (buy: do not pay above;
        sell: do not accept below).  None = no hard limit.
    adv_usd : float
        Estimated 30-day average daily volume in USD.
    sigma_daily : float
        Daily volatility fraction for impact/slippage estimation.
    asset_class : str
        "equity" or "crypto".  Drives venue selection defaults.
    """
    symbol:           str
    target_qty:       float
    urgency:          float          = 0.5
    max_slippage_bps: float          = 10.0
    time_limit_bars:  int            = 4
    price_limit:      Optional[float]= None
    adv_usd:          float          = 1_000_000.0
    sigma_daily:      float          = 0.02
    asset_class:      str            = "equity"


@dataclass
class RoutingDecision:
    """
    A single child order produced by the smart router.

    Attributes
    ----------
    venue : str             Key into VENUES.
    order_type : str        "market" | "limit" | "twap_slice".
    limit_price : float     Relevant for limit orders; 0.0 for market.
    qty : float             Signed quantity for this child order.
    bar_index : int         Which execution bar this child belongs to (0-based).
    estimated_cost_bps : float   Pre-trade cost estimate.
    reasoning : str         Human-readable rationale.
    """
    venue:               str
    order_type:          str
    limit_price:         float
    qty:                 float
    bar_index:           int     = 0
    estimated_cost_bps:  float   = 0.0
    reasoning:           str     = ""

    def to_dict(self) -> dict:
        return {
            "venue":               self.venue,
            "order_type":          self.order_type,
            "limit_price":         self.limit_price,
            "qty":                 self.qty,
            "bar_index":           self.bar_index,
            "estimated_cost_bps":  round(self.estimated_cost_bps, 4),
            "reasoning":           self.reasoning,
        }


# ---------------------------------------------------------------------------
# Quote record
# ---------------------------------------------------------------------------

@dataclass
class QuoteRecord:
    """Snapshot of a single venue quote for one symbol."""
    symbol:    str
    venue:     str
    bid:       float
    ask:       float
    bid_size:  float
    ask_size:  float
    ts:        float = field(default_factory=time.time)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def half_spread_bps(self) -> float:
        if self.mid <= 0:
            return 0.0
        return (self.spread / 2.0) / self.mid * 10_000.0


# ---------------------------------------------------------------------------
# LiquidityMap
# ---------------------------------------------------------------------------

class LiquidityMap:
    """
    In-memory map of live bid-ask quotes and depth, per symbol per venue.

    Thread-safe via RLock.
    """

    _STALE_SECS = 60.0   # quotes older than this are considered stale

    def __init__(self) -> None:
        # (symbol, venue) -> QuoteRecord
        self._quotes: dict[tuple[str, str], QuoteRecord] = {}
        self._lock   = threading.RLock()

        # Circuit-breaker state: set of venues currently disabled
        self._circuit_breakers: set[str] = set()

    # ------------------------------------------------------------------
    # Quote updates
    # ------------------------------------------------------------------

    def update_quote(
        self,
        symbol:   str,
        venue:    str,
        bid:      float,
        ask:      float,
        bid_size: float,
        ask_size: float,
        ts:       Optional[float] = None,
    ) -> None:
        """Ingest a new top-of-book quote."""
        record = QuoteRecord(
            symbol=symbol, venue=venue,
            bid=bid, ask=ask,
            bid_size=bid_size, ask_size=ask_size,
            ts=ts if ts is not None else time.time(),
        )
        with self._lock:
            self._quotes[(symbol, venue)] = record

    def get_quote(self, symbol: str, venue: str) -> Optional[QuoteRecord]:
        """Return the most recent quote for (symbol, venue), or None."""
        with self._lock:
            q = self._quotes.get((symbol, venue))
        if q is None:
            return None
        if time.time() - q.ts > self._STALE_SECS:
            return None   # treat stale as no quote
        return q

    def best_bid(self, symbol: str, venue: str) -> float:
        q = self.get_quote(symbol, venue)
        return q.bid if q else 0.0

    def best_ask(self, symbol: str, venue: str) -> float:
        q = self.get_quote(symbol, venue)
        return q.ask if q else 0.0

    # ------------------------------------------------------------------
    # Venue selection
    # ------------------------------------------------------------------

    def best_venue(
        self,
        symbol:         str,
        order_size_usd: float,
        candidates:     Optional[list[str]] = None,
        estimator:      Optional[CostEstimator] = None,
        adv_usd:        float = 1_000_000.0,
        sigma_daily:    float = 0.02,
        side:           str   = "buy",
    ) -> str:
        """
        Return the venue with the lowest estimated all-in cost for a given
        order size, considering live spreads and circuit-breaker status.

        If no live spread data is available the CostEstimator default is used.

        Parameters
        ----------
        candidates : list[str] | None
            Restrict to these venues.  None = all venues.
        """
        venues = candidates if candidates else list(VENUES.keys())

        # Exclude tripped circuit breakers
        with self._lock:
            active_cbs = set(self._circuit_breakers)
        venues = [v for v in venues if v not in active_cbs]
        if not venues:
            log.warning("All candidate venues have active circuit breakers; ignoring CBs")
            venues = candidates if candidates else list(VENUES.keys())

        if estimator is None:
            estimator = CostEstimator()

        best     = venues[0]
        best_bps = math.inf

        for v in venues:
            q = self.get_quote(symbol, v)
            half_spread = q.half_spread_bps if q else estimator.DEFAULT_HALF_SPREAD_BPS
            try:
                est = estimator.estimate(
                    symbol=symbol,
                    order_size_usd=order_size_usd,
                    side=side,
                    venue=v,
                    adv_usd=adv_usd,
                    sigma_daily=sigma_daily,
                    half_spread_bps=half_spread,
                )
                if est.total_bps < best_bps:
                    best_bps = est.total_bps
                    best     = v
            except ValueError:
                continue

        return best

    # ------------------------------------------------------------------
    # Circuit breakers
    # ------------------------------------------------------------------

    def trip_circuit_breaker(self, venue: str) -> None:
        """Mark a venue as unavailable (circuit breaker tripped)."""
        with self._lock:
            self._circuit_breakers.add(venue)
        log.warning("LiquidityMap: circuit breaker tripped for venue '%s'", venue)

    def reset_circuit_breaker(self, venue: str) -> None:
        """Re-enable a venue after a circuit breaker was tripped."""
        with self._lock:
            self._circuit_breakers.discard(venue)
        log.info("LiquidityMap: circuit breaker reset for venue '%s'", venue)

    def is_circuit_broken(self, venue: str) -> bool:
        with self._lock:
            return venue in self._circuit_breakers

    def all_quotes(self) -> list[QuoteRecord]:
        with self._lock:
            return list(self._quotes.values())


# ---------------------------------------------------------------------------
# SmartRouter
# ---------------------------------------------------------------------------

# Thresholds for urgency-based routing
_URGENCY_MARKET    = 0.8   # >= this -> market order (immediate)
_URGENCY_PATIENT   = 0.3   # <= this -> TWAP over time_limit_bars
# Large order threshold: fraction of ADV above which we split via AC schedule
_LARGE_ORDER_ADV_FRACTION = 0.01


class SmartRouter:
    """
    Smart order router.

    Routing logic
    -------------
    1. Determine order side and notional.
    2. Select primary venue via LiquidityMap.best_venue().
    3. Classify urgency:
       - urgency >= 0.8  -> single market order on primary venue
       - urgency <= 0.3  -> TWAP sliced over time_limit_bars
       - else            -> aggressive limit at mid + 0.5 * spread
    4. Large orders (> _LARGE_ORDER_ADV_FRACTION of ADV):
       - Always split into child orders using Almgren-Chriss schedule
       - One RoutingDecision per bar
    5. Post every decision to ExecutionLogger.

    Parameters
    ----------
    estimator : CostEstimator | None
        Pre-trade cost model.  Created with defaults if None.
    logger : ExecutionLogger | None
        Execution event logger.  Created with in-memory-only log if None.
    """

    def __init__(
        self,
        estimator: Optional[CostEstimator]  = None,
        logger:    Optional["ExecutionLogger"] = None,
    ) -> None:
        self._estimator = estimator or CostEstimator()
        self._logger    = logger or ExecutionLogger()

    # ------------------------------------------------------------------
    # Primary routing entry point
    # ------------------------------------------------------------------

    def route(
        self,
        intent:    OrderIntent,
        liquidity: LiquidityMap,
    ) -> list[RoutingDecision]:
        """
        Produce an execution plan for *intent* given current *liquidity*.

        Returns a list of RoutingDecision objects (one per child order /
        execution bar).  The caller is responsible for submitting them in
        the correct order (sorted by bar_index).

        Parameters
        ----------
        intent : OrderIntent
        liquidity : LiquidityMap

        Returns
        -------
        list[RoutingDecision]
        """
        side          = "buy" if intent.target_qty > 0 else "sell"
        abs_qty       = abs(intent.target_qty)
        notional_est  = abs_qty  # we treat qty as USD-normalised for simplicity

        # Determine asset-class default venues
        if intent.asset_class == "crypto":
            default_venues = ["alpaca_crypto", "binance_spot", "coinbase"]
        else:
            default_venues = ["alpaca_equity"]

        # Venue selection
        venue = liquidity.best_venue(
            symbol         = intent.symbol,
            order_size_usd = notional_est,
            candidates     = [v for v in default_venues if v in VENUES],
            estimator      = self._estimator,
            adv_usd        = intent.adv_usd,
            sigma_daily    = intent.sigma_daily,
            side           = side,
        )

        # Live quote on chosen venue
        quote = liquidity.get_quote(intent.symbol, venue)

        # Determine if this is a large order
        pct_adv = notional_est / intent.adv_usd if intent.adv_usd > 0 else 0.0
        is_large = pct_adv >= _LARGE_ORDER_ADV_FRACTION

        # ---- Route by urgency -------------------------------------------
        decisions: list[RoutingDecision] = []

        if is_large:
            decisions = self._route_large_order(intent, venue, quote)
        elif intent.urgency >= _URGENCY_MARKET:
            decisions = self._route_immediate(intent, venue, quote)
        elif intent.urgency <= _URGENCY_PATIENT:
            decisions = self._route_twap(intent, venue, quote)
        else:
            decisions = self._route_aggressive_limit(intent, venue, quote)

        # Warn if estimated cost exceeds max_slippage_bps
        total_cost = sum(d.estimated_cost_bps for d in decisions)
        if total_cost > intent.max_slippage_bps * len(decisions):
            log.warning(
                "SmartRouter: symbol=%s estimated_cost=%.2f bps > max_slippage=%.2f bps",
                intent.symbol, total_cost / max(len(decisions), 1), intent.max_slippage_bps,
            )

        # Log all decisions
        for d in decisions:
            self._logger.log_decision(intent, d)

        return decisions

    # ------------------------------------------------------------------
    # Routing strategies
    # ------------------------------------------------------------------

    def _route_immediate(
        self,
        intent: OrderIntent,
        venue:  str,
        quote:  Optional[QuoteRecord],
    ) -> list[RoutingDecision]:
        """Single market order for immediate execution."""
        signed_qty = intent.target_qty
        est = self._estimate_cost(intent, venue, quote, is_maker=False)
        return [RoutingDecision(
            venue              = venue,
            order_type         = "market",
            limit_price        = 0.0,
            qty                = signed_qty,
            bar_index          = 0,
            estimated_cost_bps = est.total_bps,
            reasoning          = f"urgency={intent.urgency:.2f} >= {_URGENCY_MARKET} -- market order",
        )]

    def _route_twap(
        self,
        intent: OrderIntent,
        venue:  str,
        quote:  Optional[QuoteRecord],
    ) -> list[RoutingDecision]:
        """Uniform TWAP slices over time_limit_bars."""
        n    = max(intent.time_limit_bars, 1)
        fracs = [1.0 / n] * n
        return self._build_schedule(intent, venue, quote, fracs, "twap")

    def _route_aggressive_limit(
        self,
        intent: OrderIntent,
        venue:  str,
        quote:  Optional[QuoteRecord],
    ) -> list[RoutingDecision]:
        """Aggressive limit at mid + 0.5 * spread (i.e., between mid and near-touch)."""
        if quote is not None and quote.mid > 0:
            mid    = quote.mid
            spread = quote.spread
            if intent.target_qty > 0:
                # Buy: bid mid + half_spread (between mid and ask)
                limit_px = mid + 0.5 * spread * 0.5
            else:
                # Sell: offer mid - half_spread (between mid and bid)
                limit_px = mid - 0.5 * spread * 0.5
        else:
            limit_px = 0.0   # no live quote; fall back to market

        order_type = "limit" if limit_px > 0 else "market"
        est = self._estimate_cost(intent, venue, quote, is_maker=(order_type == "limit"))
        return [RoutingDecision(
            venue              = venue,
            order_type         = order_type,
            limit_price        = limit_px,
            qty                = intent.target_qty,
            bar_index          = 0,
            estimated_cost_bps = est.total_bps,
            reasoning          = (
                f"urgency={intent.urgency:.2f} in ({_URGENCY_PATIENT},{_URGENCY_MARKET})"
                f" -- aggressive limit @ {limit_px:.4f}"
            ),
        )]

    def _route_large_order(
        self,
        intent: OrderIntent,
        venue:  str,
        quote:  Optional[QuoteRecord],
    ) -> list[RoutingDecision]:
        """
        Split a large order using the Almgren-Chriss optimal schedule.

        Uses intent.time_limit_bars as the number of execution intervals.
        Each child order is a limit order at the current mid (will be
        refreshed before submission in production).
        """
        n = max(intent.time_limit_bars, 2)

        bars_per_day = 26.0
        sigma_bar    = intent.sigma_daily / math.sqrt(bars_per_day)
        eta          = 0.1
        gamma        = 0.05
        lam          = 1e-6

        total_notional = abs(intent.target_qty)
        participation  = total_notional / intent.adv_usd if intent.adv_usd > 0 else 1.0

        raw_trades = _almgren_chriss_trajectory(
            total_shares = participation,
            n_bars       = n,
            sigma        = sigma_bar,
            eta          = eta,
            gamma        = gamma,
            lam          = lam,
        )

        total_raw = sum(raw_trades)
        fracs     = [t / total_raw for t in raw_trades] if total_raw > 0 else [1.0 / n] * n
        return self._build_schedule(intent, venue, quote, fracs, "ac_split")

    def _build_schedule(
        self,
        intent:     OrderIntent,
        venue:      str,
        quote:      Optional[QuoteRecord],
        fracs:      list[float],
        reason_tag: str,
    ) -> list[RoutingDecision]:
        """Convert a list of size fractions into RoutingDecision objects."""
        decisions = []
        sign      = 1.0 if intent.target_qty > 0 else -1.0
        mid_px    = quote.mid if (quote and quote.mid > 0) else 0.0

        for i, frac in enumerate(fracs):
            child_qty = sign * abs(intent.target_qty) * frac
            child_est = self._estimate_cost(intent, venue, quote, is_maker=True)
            # Scale dollar_cost by frac but keep bps the same
            decisions.append(RoutingDecision(
                venue              = venue,
                order_type         = "twap_slice" if reason_tag == "twap" else "limit",
                limit_price        = mid_px,
                qty                = child_qty,
                bar_index          = i,
                estimated_cost_bps = child_est.total_bps,
                reasoning          = (
                    f"{reason_tag} bar {i}/{len(fracs)-1}"
                    f" frac={frac:.4f} urgency={intent.urgency:.2f}"
                ),
            ))
        return decisions

    # ------------------------------------------------------------------
    # Implementation shortfall
    # ------------------------------------------------------------------

    def implementation_shortfall(
        self,
        decision_mid:  float,
        fill_price:    float,
        side:          str,
    ) -> float:
        """
        Compute implementation shortfall in bps.

        IS = (fill_price - decision_mid) / decision_mid * 10_000  (buy side)
        IS = (decision_mid - fill_price) / decision_mid * 10_000  (sell side)

        A positive IS means we paid more (bought higher / sold lower) than
        the midprice at decision time -- adverse execution.
        """
        if decision_mid <= 0:
            return 0.0
        if side == "buy":
            return (fill_price - decision_mid) / decision_mid * 10_000.0
        else:
            return (decision_mid - fill_price) / decision_mid * 10_000.0

    def record_fill(
        self,
        decision:     RoutingDecision,
        fill_price:   float,
        fill_qty:     float,
        decision_mid: float,
        side:         str,
    ) -> float:
        """
        Record an actual fill and return implementation shortfall in bps.

        The shortfall is logged to ExecutionLogger for post-trade reporting.
        """
        is_bps = self.implementation_shortfall(decision_mid, fill_price, side)
        self._logger.log_fill(
            venue       = decision.venue,
            symbol      = "",   # caller should pass symbol via subclass if needed
            qty         = fill_qty,
            fill_price  = fill_price,
            mid_at_decision = decision_mid,
            is_bps      = is_bps,
            bar_index   = decision.bar_index,
        )
        return is_bps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_cost(
        self,
        intent:   OrderIntent,
        venue:    str,
        quote:    Optional[QuoteRecord],
        is_maker: bool = False,
    ) -> CostEstimate:
        half_spread = quote.half_spread_bps if quote else None
        side        = "buy" if intent.target_qty > 0 else "sell"
        try:
            return self._estimator.estimate(
                symbol          = intent.symbol,
                order_size_usd  = abs(intent.target_qty),
                side            = side,
                venue           = venue,
                adv_usd         = intent.adv_usd,
                sigma_daily     = intent.sigma_daily,
                half_spread_bps = half_spread,
                is_maker        = is_maker,
            )
        except Exception as exc:
            log.warning("SmartRouter: cost estimate failed: %s", exc)
            from execution.cost_model import CostEstimate
            return CostEstimate(
                symbol=intent.symbol, venue=venue, side=side,
                order_size_usd=abs(intent.target_qty),
            )


# ---------------------------------------------------------------------------
# DarkPoolChecker
# ---------------------------------------------------------------------------

# Approximate threshold for mid/large cap US equities by market cap tier
_LARGE_CAP_SYMBOLS = {
    "SPY", "QQQ", "IWM", "GLD", "TLT", "SLV", "USO",
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META",
    "JPM", "BAC", "WFC", "GS", "MS",
}

# Typical dark pool fill rates by order size as % of ADV
_DARK_POOL_FILL_TABLE: list[tuple[float, float]] = [
    # (pct_adv_threshold, fill_probability)
    (0.001,  0.90),
    (0.005,  0.75),
    (0.01,   0.60),
    (0.05,   0.40),
    (0.10,   0.20),
    (1.00,   0.05),
]


class DarkPoolChecker:
    """
    Estimates dark pool availability and fill probability for equities.

    Dark pool routing is only relevant for equities (not crypto).
    Mid/large cap symbols have dark pool access by default.

    Parameters
    ----------
    eligible_symbols : set[str] | None
        Override the default large-cap set.
    """

    def __init__(self, eligible_symbols: Optional[set[str]] = None) -> None:
        self._eligible = eligible_symbols if eligible_symbols is not None else _LARGE_CAP_SYMBOLS

    def has_dark_pool(self, symbol: str) -> bool:
        """Return True if the symbol is eligible for dark pool routing."""
        return symbol in self._eligible

    def fill_probability(
        self,
        symbol:         str,
        order_size_usd: float,
        adv_usd:        float,
    ) -> float:
        """
        Estimate the probability of a complete dark-pool fill.

        Uses a step-function lookup table based on order size as a
        fraction of ADV.

        Returns
        -------
        float
            Probability in [0, 1].  0.0 for ineligible symbols.
        """
        if not self.has_dark_pool(symbol):
            return 0.0
        if adv_usd <= 0:
            return 0.0

        pct_adv = order_size_usd / adv_usd

        # Walk table and interpolate between adjacent rows
        for i, (threshold, prob) in enumerate(_DARK_POOL_FILL_TABLE):
            if pct_adv <= threshold:
                if i == 0:
                    return prob
                prev_thresh, prev_prob = _DARK_POOL_FILL_TABLE[i - 1]
                # Linear interpolation
                t = (pct_adv - prev_thresh) / (threshold - prev_thresh)
                return prev_prob + t * (prob - prev_prob)

        # Larger than largest threshold
        return _DARK_POOL_FILL_TABLE[-1][1]

    def recommend_dark_pool(
        self,
        symbol:         str,
        order_size_usd: float,
        adv_usd:        float,
        min_prob:       float = 0.5,
    ) -> bool:
        """
        Return True if dark pool routing is recommended.

        Recommended when fill_probability >= min_prob.
        """
        return self.fill_probability(symbol, order_size_usd, adv_usd) >= min_prob

    def add_eligible_symbol(self, symbol: str) -> None:
        """Manually add a symbol to the dark pool eligible set."""
        self._eligible.add(symbol)


# ---------------------------------------------------------------------------
# ExecutionLogger
# ---------------------------------------------------------------------------

_SCHEMA_DECISIONS = """
CREATE TABLE IF NOT EXISTS routing_decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    symbol        TEXT    NOT NULL,
    side          TEXT    NOT NULL,
    target_qty    REAL    NOT NULL,
    urgency       REAL    NOT NULL,
    venue         TEXT    NOT NULL,
    order_type    TEXT    NOT NULL,
    limit_price   REAL    NOT NULL,
    child_qty     REAL    NOT NULL,
    bar_index     INTEGER NOT NULL,
    estimated_bps REAL    NOT NULL,
    reasoning     TEXT    NOT NULL
)
"""

_SCHEMA_FILLS = """
CREATE TABLE IF NOT EXISTS fill_outcomes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL    NOT NULL,
    venue           TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    qty             REAL    NOT NULL,
    fill_price      REAL    NOT NULL,
    mid_at_decision REAL    NOT NULL,
    is_bps          REAL    NOT NULL,
    bar_index       INTEGER NOT NULL
)
"""

_IDX_DECISIONS = "CREATE INDEX IF NOT EXISTS idx_rd_ts     ON routing_decisions (ts)"
_IDX_FILLS     = "CREATE INDEX IF NOT EXISTS idx_fo_ts     ON fill_outcomes (ts)"
_IDX_FILLS_SYM = "CREATE INDEX IF NOT EXISTS idx_fo_symbol ON fill_outcomes (symbol)"


class ExecutionLogger:
    """
    SQLite-backed log of routing decisions and fill outcomes.

    If no db_path is provided the logger operates entirely in-memory
    using a list (no persistence).

    Methods
    -------
    log_decision(intent, decision)
        Record a routing decision.
    log_fill(venue, symbol, qty, fill_price, mid_at_decision, is_bps, bar_index)
        Record an actual fill outcome.
    get_execution_quality(symbol=None, since=None) -> dict
        Aggregate post-trade quality metrics.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path
        self._lock    = threading.RLock()

        # In-memory fallback lists
        self._decisions: list[dict] = []
        self._fills:     list[dict] = []

        if db_path is not None:
            self._init_db()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_decision(
        self,
        intent:   OrderIntent,
        decision: RoutingDecision,
    ) -> None:
        side = "buy" if intent.target_qty > 0 else "sell"
        row  = {
            "ts":            time.time(),
            "symbol":        intent.symbol,
            "side":          side,
            "target_qty":    intent.target_qty,
            "urgency":       intent.urgency,
            "venue":         decision.venue,
            "order_type":    decision.order_type,
            "limit_price":   decision.limit_price,
            "child_qty":     decision.qty,
            "bar_index":     decision.bar_index,
            "estimated_bps": decision.estimated_cost_bps,
            "reasoning":     decision.reasoning,
        }
        with self._lock:
            self._decisions.append(row)
        if self._db_path:
            self._insert("routing_decisions", row)

    def log_fill(
        self,
        venue:           str,
        symbol:          str,
        qty:             float,
        fill_price:      float,
        mid_at_decision: float,
        is_bps:          float,
        bar_index:       int,
    ) -> None:
        row = {
            "ts":              time.time(),
            "venue":           venue,
            "symbol":          symbol,
            "qty":             qty,
            "fill_price":      fill_price,
            "mid_at_decision": mid_at_decision,
            "is_bps":          is_bps,
            "bar_index":       bar_index,
        }
        with self._lock:
            self._fills.append(row)
        if self._db_path:
            self._insert("fill_outcomes", row)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_execution_quality(
        self,
        symbol: Optional[str] = None,
        since:  Optional[float] = None,
    ) -> dict:
        """
        Compute aggregate execution quality metrics from recorded fills.

        Parameters
        ----------
        symbol : str | None     Filter to this symbol if provided.
        since : float | None    Unix timestamp lower bound.  Defaults to
                                beginning of records.

        Returns
        -------
        dict with keys:
            "n_fills"         : int
            "avg_is_bps"      : float    (avg implementation shortfall)
            "median_is_bps"   : float
            "pct_positive_is" : float    (fraction with adverse IS)
            "per_venue"       : dict[venue, {avg_is_bps, n}]
        """
        with self._lock:
            fills = list(self._fills)

        if since is not None:
            fills = [f for f in fills if f["ts"] >= since]
        if symbol is not None:
            fills = [f for f in fills if f["symbol"] == symbol]

        if not fills:
            return {
                "n_fills": 0,
                "avg_is_bps": 0.0,
                "median_is_bps": 0.0,
                "pct_positive_is": 0.0,
                "per_venue": {},
            }

        is_values = [f["is_bps"] for f in fills]
        n         = len(is_values)
        avg_is    = sum(is_values) / n
        sorted_is = sorted(is_values)
        median_is = sorted_is[n // 2]
        pct_pos   = sum(1 for x in is_values if x > 0) / n

        venue_buckets: dict[str, list[float]] = defaultdict(list)
        for f in fills:
            venue_buckets[f["venue"]].append(f["is_bps"])

        per_venue = {
            v: {
                "avg_is_bps": round(sum(vals) / len(vals), 4),
                "n": len(vals),
            }
            for v, vals in venue_buckets.items()
        }

        return {
            "n_fills":          n,
            "avg_is_bps":       round(avg_is,    4),
            "median_is_bps":    round(median_is, 4),
            "pct_positive_is":  round(pct_pos,   4),
            "per_venue":        per_venue,
        }

    def decisions_since(self, since: float) -> list[dict]:
        """Return all routing decisions logged after *since* (Unix ts)."""
        with self._lock:
            return [d for d in self._decisions if d["ts"] >= since]

    def clear(self) -> None:
        """Flush in-memory logs (does not affect SQLite)."""
        with self._lock:
            self._decisions.clear()
            self._fills.clear()

    # ------------------------------------------------------------------
    # SQLite
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        assert self._db_path is not None
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(self._db_path))
        con.execute(_SCHEMA_DECISIONS)
        con.execute(_SCHEMA_FILLS)
        con.execute(_IDX_DECISIONS)
        con.execute(_IDX_FILLS)
        con.execute(_IDX_FILLS_SYM)
        con.commit()
        con.close()

    def _insert(self, table: str, row: dict) -> None:
        try:
            con = sqlite3.connect(str(self._db_path))
            cols   = ", ".join(row.keys())
            placeholders = ", ".join("?" * len(row))
            con.execute(
                f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
                list(row.values()),
            )
            con.commit()
            con.close()
        except Exception as exc:
            log.warning("ExecutionLogger: DB insert failed (%s): %s", table, exc)
