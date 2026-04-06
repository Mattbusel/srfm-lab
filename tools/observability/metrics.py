"""
tools/observability/metrics.py
================================
Prometheus metrics server for the LARSA live trader.

Exposes /metrics on :9090 via a background thread.
Provides helpers for recording fills, orders, BH state, and portfolio
snapshots so the live_trader can instrument itself with a single import.

Usage:
    from tools.observability.metrics import PrometheusMetrics

    prom = PrometheusMetrics(port=9090)
    prom.start()

    # During trading loop:
    prom.record_fill("BTC", "buy", qty=0.01, price=65000, latency_ms=34.2,
                     slippage_bps=1.8)
    prom.update_portfolio(equity=102_500, drawdown_pct=1.4,
                          buying_power=48_000, open_positions=3)
    prom.record_bh_state("BTC", "1h", active=True, mass=0.72, tf_score=3)

    prom.stop()

Dependencies:
    pip install prometheus_client
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import prometheus_client; degrade gracefully if missing.
# ---------------------------------------------------------------------------
try:
    import prometheus_client as prom_lib
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Summary,
        start_http_server,
        REGISTRY,
    )
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False
    log.warning("prometheus_client not installed — metrics will be no-ops")


# ---------------------------------------------------------------------------
# Histogram / Summary bucket config
# ---------------------------------------------------------------------------

_LATENCY_BUCKETS = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)
_NOTIONAL_BUCKETS = (10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000)
_SLIPPAGE_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250)
_HOLD_BUCKETS = (1, 5, 15, 30, 60, 120, 240, 480, 960, 1440, 2880, 5760)

# PnL per trade in dollars
_PNL_QUANTILES = {0.5: 0.05, 0.9: 0.01, 0.95: 0.005, 0.99: 0.001}


class _NoopMetric:
    """Stand-in for any Prometheus metric when the library is unavailable."""

    def labels(self, **_kw):          # type: ignore[override]
        return self

    def inc(self, _amount=1):         pass
    def dec(self, _amount=1):         pass
    def set(self, _value):            pass
    def observe(self, _value):        pass
    def time(self):                   return _NoopContext()


class _NoopContext:
    def __enter__(self):  return self
    def __exit__(self, *_): pass


class PrometheusMetrics:
    """
    Prometheus metrics server for the LARSA live trader.

    All metric families are created on a dedicated ``CollectorRegistry``
    so the server can coexist with other Prometheus users in the same
    process without double-registration errors.

    Thread safety:
        All ``record_*`` and ``update_*`` methods are safe to call from
        any thread.  The HTTP server itself runs in a daemon thread
        managed by prometheus_client.
    """

    # ------------------------------------------------------------------ init
    def __init__(self, port: int = 9090, namespace: str = "larsa") -> None:
        self._port = port
        self._ns = namespace
        self._running = False
        self._lock = threading.Lock()

        if not _PROM_AVAILABLE:
            log.warning("PrometheusMetrics: prometheus_client unavailable; "
                        "all metrics are no-ops.")
            self._init_noops()
            return

        # Use a fresh registry to avoid collisions during hot-reloads.
        self._registry = CollectorRegistry(auto_describe=True)
        self._init_metrics()

    # --------------------------------------------------------- internal setup
    def _g(self, name: str, doc: str, labelnames=()) -> "Gauge":
        if not _PROM_AVAILABLE:
            return _NoopMetric()  # type: ignore
        return prom_lib.Gauge(
            f"{self._ns}_{name}", doc,
            labelnames=labelnames,
            registry=self._registry,
        )

    def _c(self, name: str, doc: str, labelnames=()) -> "Counter":
        if not _PROM_AVAILABLE:
            return _NoopMetric()  # type: ignore
        return prom_lib.Counter(
            f"{self._ns}_{name}", doc,
            labelnames=labelnames,
            registry=self._registry,
        )

    def _h(self, name: str, doc: str, buckets, labelnames=()) -> "Histogram":
        if not _PROM_AVAILABLE:
            return _NoopMetric()  # type: ignore
        return prom_lib.Histogram(
            f"{self._ns}_{name}", doc,
            buckets=buckets,
            labelnames=labelnames,
            registry=self._registry,
        )

    def _s(self, name: str, doc: str, quantiles: dict, labelnames=()) -> "Summary":
        if not _PROM_AVAILABLE:
            return _NoopMetric()  # type: ignore
        # prometheus_client Summary accepts quantiles as list of (quantile, error)
        q_list = [(q, e) for q, e in quantiles.items()]
        return prom_lib.Summary(
            f"{self._ns}_{name}", doc,
            labelnames=labelnames,
            registry=self._registry,
        )

    def _init_metrics(self) -> None:
        """Create all metric families."""
        ns = self._ns

        # ── Portfolio gauges ──────────────────────────────────────────────────
        self.equity = self._g(
            "equity_usd",
            "Current portfolio equity in USD",
        )
        self.drawdown_pct = self._g(
            "drawdown_pct",
            "Current drawdown from high-water mark as a percentage (0-100)",
        )
        self.open_positions = self._g(
            "open_positions",
            "Number of currently open positions (equity + crypto)",
        )
        self.buying_power = self._g(
            "buying_power_usd",
            "Available buying power reported by the broker",
        )

        # ── Position fraction per symbol ──────────────────────────────────────
        self.position_frac = self._g(
            "position_frac",
            "Fraction of portfolio allocated to this symbol (0-1)",
            labelnames=["symbol"],
        )

        # ── Circuit breaker ───────────────────────────────────────────────────
        self.circuit_breaker_state = self._g(
            "circuit_breaker_state",
            "Circuit breaker state: 1=tripped (trading halted), 0=normal",
        )

        # ── Trade counters ────────────────────────────────────────────────────
        self.trades_placed = self._c(
            "trades_placed_total",
            "Total number of trade orders submitted to the broker",
            labelnames=["symbol", "side"],
        )
        self.orders_filled = self._c(
            "orders_filled_total",
            "Total number of orders that received a fill confirmation",
            labelnames=["symbol", "side"],
        )
        self.orders_rejected = self._c(
            "orders_rejected_total",
            "Total number of orders rejected by the broker",
            labelnames=["symbol", "reason"],
        )
        self.fills_total = self._c(
            "fills_total",
            "Total number of individual fill events (partial fills count separately)",
            labelnames=["symbol"],
        )

        # ── Fill / execution histograms ───────────────────────────────────────
        self.fill_latency_ms = self._h(
            "fill_latency_ms",
            "Time from order submission to first fill confirmation, in ms",
            buckets=_LATENCY_BUCKETS,
            labelnames=["symbol"],
        )
        self.order_notional = self._h(
            "order_notional_usd",
            "Order notional value in USD",
            buckets=_NOTIONAL_BUCKETS,
            labelnames=["symbol", "side"],
        )
        self.slippage_bps = self._h(
            "slippage_bps",
            "Fill slippage in basis points (|fill_px - mid_px| / mid_px * 10000)",
            buckets=_SLIPPAGE_BUCKETS,
            labelnames=["symbol"],
        )

        # ── PnL / hold-time summaries ─────────────────────────────────────────
        self.pnl_per_trade = self._s(
            "pnl_per_trade_usd",
            "Realised PnL per closed trade in USD",
            quantiles=_PNL_QUANTILES,
            labelnames=["symbol"],
        )
        self.hold_duration_minutes = self._s(
            "hold_duration_minutes",
            "Trade hold duration in minutes from entry fill to exit fill",
            quantiles=_PNL_QUANTILES,
            labelnames=["symbol"],
        )

        # ── Black-Hole (BH) specific metrics ──────────────────────────────────
        self.bh_active_count = self._g(
            "bh_active_count",
            "1 if the BH attractor is currently active for this symbol+timeframe, 0 otherwise",
            labelnames=["symbol", "timeframe"],
        )
        self.bh_mass = self._g(
            "bh_mass",
            "Black-Hole mass (normalised capture ratio) for this symbol+timeframe",
            labelnames=["symbol", "timeframe"],
        )
        self.tf_score = self._h(
            "tf_score",
            "Timeframe activation score (0-7): sum of 4×4h + 2×1h + 1×15m",
            buckets=(0, 1, 2, 3, 4, 5, 6, 7),
            labelnames=["symbol"],
        )

        # ── Process-level metadata gauge ─────────────────────────────────────
        self.strategy_version_info = self._g(
            "strategy_version_info",
            "Constant 1, labelled with the current strategy version string",
            labelnames=["version"],
        )

    def _init_noops(self) -> None:
        """Populate every attribute with a _NoopMetric when lib is missing."""
        noop = _NoopMetric()
        for attr in (
            "equity", "drawdown_pct", "open_positions", "buying_power",
            "position_frac", "circuit_breaker_state",
            "trades_placed", "orders_filled", "orders_rejected", "fills_total",
            "fill_latency_ms", "order_notional", "slippage_bps",
            "pnl_per_trade", "hold_duration_minutes",
            "bh_active_count", "bh_mass", "tf_score",
            "strategy_version_info",
        ):
            setattr(self, attr, noop)

    # ----------------------------------------------------------------- start / stop
    def start(self, version: str = "unknown") -> None:
        """
        Launch the background HTTP server serving ``/metrics``.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._running:
            return
        if not _PROM_AVAILABLE:
            log.warning("PrometheusMetrics.start(): prometheus_client not "
                        "available — /metrics endpoint will not be served.")
            self._running = True
            return

        try:
            start_http_server(self._port, registry=self._registry)
            self._running = True
            log.info("PrometheusMetrics: /metrics live on :%d", self._port)
        except OSError as exc:
            log.error("PrometheusMetrics.start() failed on port %d: %s",
                      self._port, exc)
            return

        # Mark strategy version
        try:
            self.strategy_version_info.labels(version=version).set(1)
        except Exception:
            pass

    def stop(self) -> None:
        """
        Signal shutdown.  The prometheus_client HTTP thread is a daemon
        thread and will exit automatically when the main process ends.
        This method exists so callers can record a clean shutdown.
        """
        self._running = False
        log.info("PrometheusMetrics: stopped (port %d)", self._port)

    # ----------------------------------------------------------------- helpers

    def record_fill(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        latency_ms: float,
        slippage_bps: float = 0.0,
        pnl: Optional[float] = None,
        hold_minutes: Optional[float] = None,
    ) -> None:
        """
        Record a completed fill event.

        Parameters
        ----------
        symbol:
            Instrument symbol, e.g. ``"BTC"`` or ``"SPY"``.
        side:
            ``"buy"`` or ``"sell"``.
        qty:
            Quantity filled (units, not notional).
        price:
            Fill price per unit.
        latency_ms:
            Milliseconds from order submission to this fill confirmation.
        slippage_bps:
            Signed slippage in basis points  (positive = paid more than mid).
        pnl:
            Realised PnL in USD if this fill closes a position.
        hold_minutes:
            Hold duration in minutes if this fill closes a position.
        """
        notional = abs(qty * price)
        sym = symbol.upper()
        side = side.lower()

        try:
            self.orders_filled.labels(symbol=sym, side=side).inc()
            self.fills_total.labels(symbol=sym).inc()
            self.fill_latency_ms.labels(symbol=sym).observe(latency_ms)
            self.order_notional.labels(symbol=sym, side=side).observe(notional)
            self.slippage_bps.labels(symbol=sym).observe(abs(slippage_bps))

            if pnl is not None:
                self.pnl_per_trade.labels(symbol=sym).observe(pnl)
            if hold_minutes is not None:
                self.hold_duration_minutes.labels(symbol=sym).observe(hold_minutes)
        except Exception as exc:
            log.debug("PrometheusMetrics.record_fill error: %s", exc)

    def record_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        rejected: bool = False,
        reject_reason: str = "unknown",
    ) -> None:
        """
        Record an order submission attempt.

        Parameters
        ----------
        symbol, side, qty, price:
            Order details.
        rejected:
            If True, increments ``orders_rejected`` instead of ``trades_placed``.
        reject_reason:
            Short reason string used as the ``reason`` label on rejection.
        """
        sym = symbol.upper()
        side = side.lower()
        notional = abs(qty * price)

        try:
            if rejected:
                self.orders_rejected.labels(
                    symbol=sym, reason=reject_reason
                ).inc()
            else:
                self.trades_placed.labels(symbol=sym, side=side).inc()
                self.order_notional.labels(symbol=sym, side=side).observe(notional)
        except Exception as exc:
            log.debug("PrometheusMetrics.record_order error: %s", exc)

    def record_bh_state(
        self,
        symbol: str,
        timeframe: str,
        active: bool,
        mass: float,
        tf_score: Optional[int] = None,
    ) -> None:
        """
        Update Black-Hole state metrics for a symbol+timeframe combination.

        Parameters
        ----------
        symbol:
            Instrument symbol, e.g. ``"BTC"``.
        timeframe:
            One of ``"15m"``, ``"1h"``, ``"4h"``.
        active:
            Whether the BH attractor is currently active.
        mass:
            Normalised BH mass (capture ratio), typically 0-1.
        tf_score:
            Combined timeframe score (0-7).  Only observed when ``timeframe``
            is the "primary" timeframe for this update (i.e. ``"1h"``).
        """
        sym = symbol.upper()
        try:
            self.bh_active_count.labels(symbol=sym, timeframe=timeframe).set(
                1 if active else 0
            )
            self.bh_mass.labels(symbol=sym, timeframe=timeframe).set(mass)
            if tf_score is not None:
                self.tf_score.labels(symbol=sym).observe(tf_score)
        except Exception as exc:
            log.debug("PrometheusMetrics.record_bh_state error: %s", exc)

    def update_portfolio(
        self,
        equity: float,
        drawdown_pct: float,
        buying_power: float,
        open_positions: int,
        circuit_breaker: bool = False,
        position_fracs: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Snapshot the current portfolio state.

        Parameters
        ----------
        equity:
            Total portfolio equity in USD.
        drawdown_pct:
            Current drawdown from high-water mark, as a percentage 0-100.
        buying_power:
            Available buying power from the broker.
        open_positions:
            Number of open positions (equity + crypto).
        circuit_breaker:
            True if the drawdown circuit breaker has tripped.
        position_fracs:
            Optional dict mapping symbol → fraction of portfolio, e.g.
            ``{"BTC": 0.12, "SPY": 0.08}``.
        """
        try:
            self.equity.set(equity)
            self.drawdown_pct.set(drawdown_pct)
            self.buying_power.set(buying_power)
            self.open_positions.set(open_positions)
            self.circuit_breaker_state.set(1 if circuit_breaker else 0)

            if position_fracs:
                for sym, frac in position_fracs.items():
                    self.position_frac.labels(symbol=sym.upper()).set(frac)
        except Exception as exc:
            log.debug("PrometheusMetrics.update_portfolio error: %s", exc)

    # ----------------------------------------------------------------- props
    @property
    def port(self) -> int:
        return self._port

    @property
    def running(self) -> bool:
        return self._running

    # ----------------------------------------------------------------- context manager
    def __enter__(self) -> "PrometheusMetrics":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def __repr__(self) -> str:
        status = "running" if self._running else "stopped"
        return f"<PrometheusMetrics port={self._port} status={status}>"


# ---------------------------------------------------------------------------
# Module-level singleton for convenience
# ---------------------------------------------------------------------------

_default_instance: Optional[PrometheusMetrics] = None


def get_metrics(port: int = 9090) -> PrometheusMetrics:
    """
    Return (and lazily create) the module-level singleton ``PrometheusMetrics``
    instance.  Useful for getting the same instance across multiple modules
    without passing it explicitly.

    Example::

        from tools.observability.metrics import get_metrics

        prom = get_metrics()   # creates on first call, returns same object after
    """
    global _default_instance
    if _default_instance is None:
        _default_instance = PrometheusMetrics(port=port)
    return _default_instance
