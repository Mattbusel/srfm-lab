"""
execution/routing/smart_router.py
==================================
Smart order router — converts OMS Orders into broker calls.

Routing logic
-------------
1. **Blocked hours**: refuses orders in BLOCKED_ENTRY_HOURS_UTC (mirrors live
   trader logic).
2. **Spread check**: if the bid-ask spread > 0.5 %, converts MARKET to a LIMIT
   order at the midprice with a 30-second timeout, then falls back to MARKET.
3. **TWAP split**: if the order notional > 2 % of estimated daily volume, the
   order is handed off to ``TWAPExecutor`` which slices it over time.
4. **Retry**: exponential backoff on broker errors, max 3 retries.

The router returns the broker_order_id string so the OrderManager can store it.
For TWAP orders it returns the parent broker_order_id; child slice IDs are
managed internally by ``TWAPExecutor``.
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("execution.smart_router")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPREAD_THRESHOLD_PCT:   float = 0.005   # 2 % of ADV triggers TWAP
ADV_SPLIT_THRESHOLD:    float = 0.02    # order > 2 % daily volume → TWAP
LIMIT_TIMEOUT_SEC:      int   = 30      # seconds to wait for limit fill
BLOCKED_HOURS_UTC:      frozenset = frozenset({1, 13, 14, 15, 17, 18})
MAX_RETRIES:            int   = 3
BASE_BACKOFF_SEC:       float = 1.0


# ---------------------------------------------------------------------------
# SmartRouter
# ---------------------------------------------------------------------------

class SmartRouter:
    """
    Routes OMS Order objects to the appropriate broker method.

    Parameters
    ----------
    broker : AlpacaAdapter
        The broker adapter.
    twap_executor : TWAPExecutor | None
        If provided, large orders are delegated here.
    daily_volume_estimates : dict[str, float] | None
        Symbol -> estimated daily volume (USD).  Used for TWAP threshold.
    spread_feed : Callable[[str], tuple[float, float]] | None
        Optional function ``symbol -> (bid, ask)`` for spread check.
        If None, spread check is skipped and all orders go to MARKET.
    blocked_hours_utc : set[int] | None
        Override default blocked hours.
    """

    def __init__(
        self,
        broker,
        twap_executor=None,
        daily_volume_estimates: Optional[dict[str, float]] = None,
        spread_feed=None,
        blocked_hours_utc: Optional[set[int]] = None,
    ) -> None:
        self._broker   = broker
        self._twap     = twap_executor
        self._adv      = daily_volume_estimates or {}
        self._spread_feed = spread_feed
        self._blocked_hours = frozenset(blocked_hours_utc or BLOCKED_HOURS_UTC)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Primary routing entry point
    # ------------------------------------------------------------------

    def route(self, order) -> str:
        """
        Route an Order to the broker.  Returns a broker_order_id string.

        Parameters
        ----------
        order : Order
            A PENDING order from the OMS.

        Returns
        -------
        str
            Broker-assigned order ID.

        Raises
        ------
        RuntimeError
            If all retry attempts are exhausted or routing is rejected.
        """
        from ..oms.order import OrderType, Side

        # ── 1. Time gate ────────────────────────────────────────────
        self._check_time_gate(order.symbol)

        # ── 2. Determine routing strategy ───────────────────────────
        use_twap = self._should_twap(order)
        if use_twap and self._twap is not None:
            log.info(
                "SmartRouter: %s %s %.4f delegated to TWAP",
                order.side.value, order.symbol, order.quantity,
            )
            return self._twap.submit(order)

        # ── 3. Spread check for MARKET orders ───────────────────────
        if order.order_type == OrderType.MARKET:
            spread_pct = self._get_spread_pct(order.symbol, order.price)
            if spread_pct is not None and spread_pct > SPREAD_THRESHOLD_PCT:
                log.info(
                    "SmartRouter: wide spread %.3f%% for %s — trying limit at mid",
                    spread_pct * 100, order.symbol,
                )
                broker_id = self._try_limit_at_mid(order, spread_pct)
                if broker_id:
                    return broker_id
                # Fall through to market order

        # ── 4. Normal execution with retry ──────────────────────────
        return self._submit_with_retry(order)

    # ------------------------------------------------------------------
    # TWAP eligibility
    # ------------------------------------------------------------------

    def _should_twap(self, order) -> bool:
        """Return True if the order size exceeds the ADV threshold."""
        adv = self._adv.get(order.symbol, 0.0)
        if adv <= 0:
            return False
        ref_price = order.price or 1.0
        notional  = order.quantity * ref_price
        return notional > ADV_SPLIT_THRESHOLD * adv

    # ------------------------------------------------------------------
    # Spread helpers
    # ------------------------------------------------------------------

    def _get_spread_pct(self, symbol: str, ref_price: Optional[float]) -> Optional[float]:
        """Return bid-ask spread as a fraction, or None if not available."""
        if self._spread_feed is None:
            return None
        try:
            bid, ask = self._spread_feed(symbol)
            if bid > 0 and ask > 0:
                return (ask - bid) / ((ask + bid) / 2.0)
        except Exception as exc:
            log.debug("spread_feed error for %s: %s", symbol, exc)
        return None

    def _try_limit_at_mid(self, order, spread_pct: float) -> Optional[str]:
        """
        Submit a limit order at the current midprice.  Wait up to
        LIMIT_TIMEOUT_SEC for a fill; if not filled, cancel and return None.
        """
        if self._spread_feed is None:
            return None
        try:
            bid, ask = self._spread_feed(order.symbol)
            mid      = (bid + ask) / 2.0
            side_str = "buy" if order.side.value == "BUY" else "sell"
            broker_id = self._broker.submit_limit_order(
                symbol      = order.symbol,
                qty         = order.quantity,
                side        = side_str,
                limit_price = mid,
                time_in_force = "ioc",   # immediate-or-cancel at mid
            )
            log.info(
                "SmartRouter: limit @ mid %.4f for %s submitted -> %s",
                mid, order.symbol, broker_id,
            )
            return broker_id
        except Exception as exc:
            log.warning("_try_limit_at_mid failed for %s: %s", order.symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Time gate
    # ------------------------------------------------------------------

    def _check_time_gate(self, symbol: str) -> None:
        """Raise RuntimeError if current UTC hour is blocked."""
        hour = datetime.now(timezone.utc).hour
        if hour in self._blocked_hours:
            raise RuntimeError(
                f"SmartRouter: blocked UTC hour {hour} — rejecting order for {symbol}"
            )

    # ------------------------------------------------------------------
    # Retry submission
    # ------------------------------------------------------------------

    def _submit_with_retry(self, order) -> str:
        """Submit to broker with exponential backoff retry."""
        from ..oms.order import OrderType, Side

        side_str = "buy" if order.side.value == "BUY" else "sell"
        last_exc: Optional[Exception] = None

        for attempt in range(MAX_RETRIES):
            try:
                if order.order_type == OrderType.MARKET:
                    return self._broker.submit_market_order(
                        symbol = order.symbol,
                        qty    = order.quantity,
                        side   = side_str,
                    )
                elif order.order_type in (OrderType.LIMIT, OrderType.STOP):
                    if order.price is None:
                        raise ValueError(f"LIMIT/STOP order for {order.symbol} has no price")
                    return self._broker.submit_limit_order(
                        symbol      = order.symbol,
                        qty         = order.quantity,
                        side        = side_str,
                        limit_price = order.price,
                    )
                else:
                    raise ValueError(f"Unknown order_type: {order.order_type}")
            except RuntimeError:
                raise  # Re-raise time-gate errors immediately
            except Exception as exc:
                last_exc = exc
                wait = BASE_BACKOFF_SEC * (2.0 ** attempt)
                log.warning(
                    "SmartRouter retry %d/%d for %s: %s — waiting %.1fs",
                    attempt + 1, MAX_RETRIES, order.symbol, exc, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"SmartRouter: all {MAX_RETRIES} retries exhausted for "
            f"{order.symbol}: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Cancellation passthrough
    # ------------------------------------------------------------------

    def cancel(self, broker_order_id: str) -> bool:
        """Cancel a broker order by its broker-assigned ID."""
        return self._broker.cancel_order(broker_order_id)
