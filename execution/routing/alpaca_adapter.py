"""
execution/routing/alpaca_adapter.py
=====================================
Alpaca broker adapter with retry logic and paper/live mode toggle.

All broker I/O is funnelled through this class so that:
- Retry/backoff is handled in one place.
- Paper vs live endpoints are switched by a single flag.
- Unit tests can mock this class without touching alpaca-py.

Dependencies
------------
    pip install alpaca-py

Credentials are loaded from environment variables:
    ALPACA_API_KEY
    ALPACA_SECRET_KEY

The adapter is intentionally *synchronous*: it wraps the alpaca-py REST
client.  The async TWAP executor calls submit_* in a thread pool via
``asyncio.get_event_loop().run_in_executor``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("execution.alpaca_adapter")

# Retry configuration
_MAX_RETRIES   = 3
_BASE_BACKOFF  = 1.0   # seconds
_BACKOFF_MULT  = 2.0   # exponential factor


# ---------------------------------------------------------------------------
# AccountInfo value object
# ---------------------------------------------------------------------------

@dataclass
class AccountInfo:
    """Snapshot of account financials from the broker."""
    equity:          float
    buying_power:    float
    portfolio_value: float
    cash:            float
    currency:        str = "USD"


# ---------------------------------------------------------------------------
# AlpacaAdapter
# ---------------------------------------------------------------------------

class AlpacaAdapter:
    """
    Thin wrapper around alpaca-py REST calls with retry and mode toggle.

    Parameters
    ----------
    api_key : str | None
        Alpaca API key.  Reads ``ALPACA_API_KEY`` env var if None.
    secret_key : str | None
        Alpaca secret.  Reads ``ALPACA_SECRET_KEY`` env var if None.
    paper : bool
        True → paper trading endpoint, False → live.
    max_retries : int
        Number of retry attempts on transient broker errors.

    Raises
    ------
    ImportError
        If ``alpaca-py`` (``alpaca.trading``) is not installed.
    """

    def __init__(
        self,
        api_key:    Optional[str] = None,
        secret_key: Optional[str] = None,
        paper:      bool          = True,
        max_retries: int          = _MAX_RETRIES,
    ) -> None:
        self._api_key    = api_key    or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self._paper      = paper
        self._max_retries = max_retries
        self._client     = self._build_client()
        log.info(
            "AlpacaAdapter init: paper=%s, key=%s...",
            paper, self._api_key[:6] if self._api_key else "MISSING",
        )

    # ------------------------------------------------------------------
    # Client construction
    # ------------------------------------------------------------------

    def _build_client(self):
        """Instantiate the alpaca-py TradingClient."""
        try:
            from alpaca.trading.client import TradingClient
            return TradingClient(
                api_key    = self._api_key,
                secret_key = self._secret_key,
                paper      = self._paper,
            )
        except ImportError as exc:
            raise ImportError(
                "alpaca-py is not installed. Run: pip install alpaca-py"
            ) from exc

    # ------------------------------------------------------------------
    # Retry helper
    # ------------------------------------------------------------------

    def _with_retry(self, fn, *args, **kwargs):
        """
        Call *fn* with exponential backoff on ``Exception``.

        Returns the function's return value, or raises the last exception
        after exhausting retries.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                wait = _BASE_BACKOFF * (_BACKOFF_MULT ** attempt)
                log.warning(
                    "Broker error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, self._max_retries, exc, wait,
                )
                time.sleep(wait)
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_market_order(
        self,
        symbol: str,
        qty:    float,
        side:   str,         # "buy" or "sell"
    ) -> str:
        """
        Submit a fractional market order.

        Parameters
        ----------
        symbol : str
            Alpaca ticker (e.g. ``"BTC/USD"`` or ``"SPY"``).
        qty : float
            Quantity in base units.
        side : str
            ``"buy"`` or ``"sell"``.

        Returns
        -------
        str
            Alpaca-assigned order ID.
        """
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        req = MarketOrderRequest(
            symbol        = symbol,
            qty           = qty,
            side          = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force = TimeInForce.GTC,
        )
        order = self._with_retry(self._client.submit_order, req)
        log.info("submit_market_order: %s %s %.6f -> broker_id=%s", side, symbol, qty, order.id)
        return str(order.id)

    def submit_limit_order(
        self,
        symbol:      str,
        qty:         float,
        side:        str,
        limit_price: float,
        time_in_force: str = "gtc",
    ) -> str:
        """
        Submit a limit order.

        Parameters
        ----------
        symbol : str
        qty : float
        side : str
            ``"buy"`` or ``"sell"``.
        limit_price : float
        time_in_force : str
            ``"gtc"`` (default) or ``"ioc"`` / ``"day"``.

        Returns
        -------
        str
            Alpaca order ID.
        """
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        _tif_map = {
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "day": TimeInForce.DAY,
        }
        req = LimitOrderRequest(
            symbol        = symbol,
            qty           = qty,
            side          = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            limit_price   = round(limit_price, 8),
            time_in_force = _tif_map.get(time_in_force.lower(), TimeInForce.GTC),
        )
        order = self._with_retry(self._client.submit_order, req)
        log.info(
            "submit_limit_order: %s %s %.6f @ %.4f -> %s",
            side, symbol, qty, limit_price, order.id,
        )
        return str(order.id)

    def cancel_order(self, alpaca_order_id: str) -> bool:
        """
        Cancel an open order by its Alpaca order ID.

        Returns True on success, False if already terminal.
        """
        try:
            self._with_retry(self._client.cancel_order_by_id, alpaca_order_id)
            log.info("cancel_order: cancelled %s", alpaca_order_id)
            return True
        except Exception as exc:
            log.warning("cancel_order: %s — %s", alpaca_order_id, exc)
            return False

    # ------------------------------------------------------------------
    # Position / account queries
    # ------------------------------------------------------------------

    def get_positions(self) -> dict[str, float]:
        """
        Return broker-reported positions as {symbol: qty}.

        Quantity is always positive (we track direction in the OMS).
        """
        raw = self._with_retry(self._client.get_all_positions)
        result: dict[str, float] = {}
        for p in raw:
            result[p.symbol] = float(p.qty)
        return result

    def get_account(self) -> AccountInfo:
        """Return a snapshot of account financials."""
        acct = self._with_retry(self._client.get_account)
        return AccountInfo(
            equity          = float(acct.equity),
            buying_power    = float(acct.buying_power),
            portfolio_value = float(acct.portfolio_value),
            cash            = float(acct.cash),
            currency        = str(getattr(acct, "currency", "USD")),
        )

    def get_order_status(self, alpaca_order_id: str) -> Optional[dict]:
        """
        Fetch current status of an order from Alpaca.

        Returns a dict with keys: ``status``, ``filled_qty``, ``filled_avg_price``.
        Returns None on error.
        """
        try:
            o = self._with_retry(self._client.get_order_by_id, alpaca_order_id)
            return {
                "status":           str(o.status),
                "filled_qty":       float(o.filled_qty or 0),
                "filled_avg_price": float(o.filled_avg_price or 0),
            }
        except Exception as exc:
            log.warning("get_order_status(%s): %s", alpaca_order_id, exc)
            return None

    # ------------------------------------------------------------------
    # Mode toggle
    # ------------------------------------------------------------------

    def switch_to_live(self) -> None:
        """Switch from paper to live endpoint.  Use with extreme caution."""
        log.warning("AlpacaAdapter: switching to LIVE trading endpoint!")
        self._paper  = False
        self._client = self._build_client()

    def switch_to_paper(self) -> None:
        """Switch back to paper endpoint."""
        log.info("AlpacaAdapter: switching to PAPER trading endpoint.")
        self._paper  = True
        self._client = self._build_client()

    @property
    def is_paper(self) -> bool:
        return self._paper
