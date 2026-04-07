"""
metrics_collector.py -- Prometheus-compatible metrics collector for SRFM live trading system.

Scrapes all microservices on a 15-second cycle and exposes /metrics at :9090.

Components scraped:
    LiveTraderScraper   -- reads state from SRFM SQLite (larsa_trades.db)
    RiskAPIScraper      -- polls FastAPI risk API at :8791
    CoordinationScraper -- polls Elixir coordination layer at :8781
    SignalEngineScraper -- reads IPC ring buffer stats

Usage:
    from infra.observability.metrics_collector import MetricsCollector
    collector = MetricsCollector()
    collector.start()          # launches HTTP server + background threads
    # ...
    collector.stop()
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("srfm.metrics_collector")

# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------
try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False
    log.warning("requests not installed -- HTTP scrapers disabled")

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Summary,
        CONTENT_TYPE_LATEST,
        generate_latest,
    )
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False
    log.warning("prometheus_client not installed -- metrics endpoint will return empty")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROMETHEUS_PORT      = int(os.environ.get("SRFM_PROM_PORT",         "9090"))
SCRAPE_INTERVAL_S    = int(os.environ.get("SRFM_SCRAPE_INTERVAL",   "15"))
RISK_API_URL         = os.environ.get("SRFM_RISK_API_URL",          "http://localhost:8791")
COORD_URL            = os.environ.get("SRFM_COORD_URL",             "http://localhost:8781")
TRADE_DB_PATH        = os.environ.get("SRFM_TRADE_DB",              "data/larsa_trades.db")
HTTP_TIMEOUT_S       = float(os.environ.get("SRFM_SCRAPE_TIMEOUT",  "5"))
MAX_SCRAPER_WORKERS  = int(os.environ.get("SRFM_SCRAPER_WORKERS",   "4"))


# ---------------------------------------------------------------------------
# Registry + metric definitions
# ---------------------------------------------------------------------------

class _PromMetrics:
    """All Prometheus metric objects on one isolated registry."""

    def __init__(self) -> None:
        if not _PROM_AVAILABLE:
            return
        self.registry = CollectorRegistry()
        R = self.registry

        # -- Gauges: portfolio-level -------------------------------------------
        self.portfolio_equity = Gauge(
            "srfm_portfolio_equity",
            "Total portfolio equity in USD",
            registry=R,
        )
        self.position_size_pct = Gauge(
            "srfm_position_size_pct",
            "Position size as percentage of equity per symbol",
            ["symbol"],
            registry=R,
        )
        self.bh_mass = Gauge(
            "srfm_bh_mass",
            "Black-Hole mass indicator per symbol and timeframe",
            ["symbol", "timeframe"],
            registry=R,
        )
        self.garch_vol = Gauge(
            "srfm_garch_vol",
            "GARCH conditional volatility forecast per symbol",
            ["symbol"],
            registry=R,
        )
        self.hurst_h = Gauge(
            "srfm_hurst_h",
            "Hurst exponent per symbol (0.5=random walk, >0.5=trending)",
            ["symbol"],
            registry=R,
        )
        self.nav_omega = Gauge(
            "srfm_nav_omega",
            "NAV omega -- angular velocity of equity geodesic",
            registry=R,
        )
        self.nav_geodesic_deviation = Gauge(
            "srfm_nav_geodesic_deviation",
            "NAV geodesic deviation from expected equity path",
            registry=R,
        )
        self.drawdown = Gauge(
            "srfm_drawdown",
            "Current drawdown from peak equity [0, 1]",
            registry=R,
        )
        self.service_up = Gauge(
            "srfm_service_up",
            "1 if microservice health check passing, 0 otherwise",
            ["service"],
            registry=R,
        )
        self.circuit_breaker_open = Gauge(
            "srfm_circuit_breaker_open",
            "1 if circuit breaker open, 0 if closed",
            ["venue"],
            registry=R,
        )
        self.var_95 = Gauge(
            "srfm_var_95",
            "95% Value-at-Risk estimate in USD",
            registry=R,
        )
        self.amihud_illiquidity = Gauge(
            "srfm_amihud_illiquidity",
            "Amihud illiquidity ratio per symbol",
            ["symbol"],
            registry=R,
        )
        self.avg_pairwise_correlation = Gauge(
            "srfm_avg_pairwise_correlation",
            "Average pairwise return correlation across portfolio",
            registry=R,
        )
        self.rolling_sharpe_30d = Gauge(
            "srfm_rolling_sharpe_30d",
            "Rolling 30-day Sharpe ratio",
            registry=R,
        )
        self.rolling_sharpe_90d = Gauge(
            "srfm_rolling_sharpe_90d",
            "Rolling 90-day Sharpe ratio",
            registry=R,
        )

        # -- Counters ----------------------------------------------------------
        self.trades_total = Counter(
            "srfm_trades_total",
            "Total number of trades executed since start",
            registry=R,
        )
        self.bars_processed = Counter(
            "srfm_bars_processed_total",
            "Total OHLCV bars processed by the signal engine",
            ["symbol", "timeframe"],
            registry=R,
        )
        self.signals_generated = Counter(
            "srfm_signals_generated_total",
            "Total trading signals generated",
            ["symbol", "direction"],
            registry=R,
        )
        self.api_calls_total = Counter(
            "srfm_api_calls_total",
            "Total external API calls made",
            ["service", "endpoint"],
            registry=R,
        )
        self.errors_total = Counter(
            "srfm_errors_total",
            "Total errors encountered",
            ["component", "error_type"],
            registry=R,
        )

        # -- Histograms --------------------------------------------------------
        self.bar_processing_latency_ms = Histogram(
            "srfm_bar_processing_latency_ms",
            "Time taken to process one OHLCV bar through the signal pipeline",
            ["symbol"],
            buckets=[0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000],
            registry=R,
        )
        self.order_execution_latency_ms = Histogram(
            "srfm_order_execution_latency_ms",
            "Time from signal to order acknowledgement in milliseconds",
            ["venue"],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
            registry=R,
        )
        self.signal_computation_ns = Histogram(
            "srfm_signal_computation_ns",
            "Signal computation time in nanoseconds",
            ["signal_type"],
            buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
            registry=R,
        )

        # -- Summaries (used for rolling Sharpe exposed as gauge above) --------
        self.scrape_duration_s = Summary(
            "srfm_scrape_duration_seconds",
            "Time taken per scrape cycle",
            ["scraper"],
            registry=R,
        )

    def render(self) -> bytes:
        if not _PROM_AVAILABLE:
            return b"# prometheus_client not installed\n"
        return generate_latest(self.registry)


# ---------------------------------------------------------------------------
# Individual scrapers
# ---------------------------------------------------------------------------

class _BaseScraper:
    """Base class for all component scrapers."""

    name: str = "base"

    def __init__(self, metrics: _PromMetrics) -> None:
        self._m = metrics
        self._last_error: Optional[str] = None

    def scrape(self) -> Dict[str, Any]:
        """Override to perform scraping. Return a dict of collected data."""
        raise NotImplementedError

    def _record_error(self, component: str, error_type: str, msg: str) -> None:
        self._last_error = msg
        log.warning(f"[{self.name}] {error_type}: {msg}")
        if _PROM_AVAILABLE:
            try:
                self._m.errors_total.labels(
                    component=component, error_type=error_type
                ).inc()
            except Exception:
                pass


class LiveTraderScraper(_BaseScraper):
    """
    Reads live trader state directly from the SRFM SQLite database.

    Populates:
        - portfolio_equity
        - position_size_pct per symbol
        - bh_mass per symbol+timeframe
        - garch_vol, hurst_h per symbol
        - drawdown
        - trades_total
        - rolling_sharpe_30d, rolling_sharpe_90d
    """

    name = "live_trader"

    def __init__(self, metrics: _PromMetrics, db_path: str = TRADE_DB_PATH) -> None:
        super().__init__(metrics)
        self._db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._prev_trade_count = 0

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if not self._db_path.exists():
                raise FileNotFoundError(f"Trade DB not found: {self._db_path}")
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA query_only=ON")
            conn.row_factory = sqlite3.Row
            self._conn = conn
        return self._conn

    def _query(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        try:
            conn = self._get_conn()
            with self._lock:
                return conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            # Schema might not exist yet -- that is fine
            if "no such table" in str(exc):
                return []
            raise

    def scrape(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if not _PROM_AVAILABLE:
            return result

        try:
            # -- Equity and drawdown from equity_snapshots --------------------
            rows = self._query(
                "SELECT equity FROM equity_snapshots ORDER BY ts DESC LIMIT 1"
            )
            if rows:
                equity = float(rows[0]["equity"])
                self._m.portfolio_equity.set(equity)
                result["equity"] = equity

            # Peak equity for drawdown
            rows = self._query(
                "SELECT MAX(equity) AS peak FROM equity_snapshots"
            )
            if rows and rows[0]["peak"] is not None:
                peak = float(rows[0]["peak"])
                if "equity" in result and peak > 0:
                    dd = max(0.0, (peak - result["equity"]) / peak)
                    self._m.drawdown.set(dd)
                    result["drawdown"] = dd

            # -- Trade count (monotonic counter) ------------------------------
            rows = self._query("SELECT COUNT(*) AS n FROM trades")
            if rows:
                n = int(rows[0]["n"])
                delta = max(0, n - self._prev_trade_count)
                if delta > 0:
                    self._m.trades_total.inc(delta)
                self._prev_trade_count = n
                result["trade_count"] = n

            # -- Per-symbol position sizes ------------------------------------
            rows = self._query(
                """
                SELECT sym, qty, current_price
                FROM   positions
                """
            )
            equity_val = result.get("equity", 1.0) or 1.0
            for row in rows:
                sym = row["sym"]
                qty = float(row["qty"] or 0)
                price = float(row["current_price"] or 0)
                pct = abs(qty * price) / equity_val * 100.0
                self._m.position_size_pct.labels(symbol=sym).set(pct)

            # -- BH mass from regime_log (latest per symbol) ------------------
            rows = self._query(
                """
                SELECT   symbol,
                         d_bh_mass, h_bh_mass, m15_bh_mass,
                         garch_vol,
                         MAX(ts) AS latest
                FROM     regime_log
                GROUP BY symbol
                """
            )
            for row in rows:
                sym = row["symbol"]
                d_mass = float(row["d_bh_mass"] or 0)
                h_mass = float(row["h_bh_mass"] or 0)
                m_mass = float(row["m15_bh_mass"] or 0)
                self._m.bh_mass.labels(symbol=sym, timeframe="daily").set(d_mass)
                self._m.bh_mass.labels(symbol=sym, timeframe="hourly").set(h_mass)
                self._m.bh_mass.labels(symbol=sym, timeframe="15m").set(m_mass)
                gv = row["garch_vol"]
                if gv is not None:
                    self._m.garch_vol.labels(symbol=sym).set(float(gv))

            # -- Rolling Sharpe (approx from recent trades) -------------------
            rows_30 = self._query(
                """
                SELECT pnl FROM trades
                WHERE ts >= datetime('now', '-30 days')
                ORDER BY ts
                """
            )
            rows_90 = self._query(
                """
                SELECT pnl FROM trades
                WHERE ts >= datetime('now', '-90 days')
                ORDER BY ts
                """
            )
            for rows_set, gauge in [
                (rows_30, self._m.rolling_sharpe_30d),
                (rows_90, self._m.rolling_sharpe_90d),
            ]:
                if len(rows_set) >= 2:
                    import math
                    import statistics
                    pnls = [float(r["pnl"]) for r in rows_set]
                    avg = statistics.mean(pnls)
                    std = statistics.stdev(pnls)
                    if std > 1e-9:
                        sharpe = avg / std * math.sqrt(252 * 8)
                        gauge.set(sharpe)

            self._m.service_up.labels(service="live_trader").set(1)
            result["status"] = "ok"

        except FileNotFoundError:
            # DB does not exist yet -- live trader not started
            self._m.service_up.labels(service="live_trader").set(0)
            result["status"] = "db_missing"
        except Exception as exc:
            self._record_error("live_trader", "scrape_error", str(exc))
            self._m.service_up.labels(service="live_trader").set(0)
            result["status"] = "error"
            result["error"] = str(exc)

        return result


class RiskAPIScraper(_BaseScraper):
    """
    Polls the FastAPI risk API at :8791.

    Expected endpoints:
        GET /health         -> {"status": "ok"}
        GET /risk/snapshot  -> {"var_95": float, "portfolio_beta": float,
                                 "avg_correlation": float,
                                 "circuit_breakers": {"alpaca": bool, "binance": bool},
                                 "amihud": {symbol: float}}
    """

    name = "risk_api"

    def __init__(self, metrics: _PromMetrics, base_url: str = RISK_API_URL) -> None:
        super().__init__(metrics)
        self._base = base_url.rstrip("/")
        self._session: Optional[Any] = None

    def _sess(self):
        if not _REQUESTS_AVAILABLE:
            return None
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers["User-Agent"] = "srfm-metrics-collector/1.0"
        return self._session

    def _get(self, path: str) -> Optional[dict]:
        sess = self._sess()
        if sess is None:
            return None
        try:
            resp = sess.get(f"{self._base}{path}", timeout=HTTP_TIMEOUT_S)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            self._record_error("risk_api", "http_error", f"{path}: {exc}")
            return None

    def scrape(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if not _PROM_AVAILABLE:
            return result

        # Health check
        health = self._get("/health")
        if health and health.get("status") == "ok":
            self._m.service_up.labels(service="risk_api").set(1)
        else:
            self._m.service_up.labels(service="risk_api").set(0)
            result["status"] = "down"
            return result

        # Risk snapshot
        snap = self._get("/risk/snapshot")
        if snap is None:
            snap = {}

        var95 = snap.get("var_95")
        if var95 is not None:
            self._m.var_95.set(float(var95))
            result["var_95"] = var95

        avg_corr = snap.get("avg_correlation")
        if avg_corr is not None:
            self._m.avg_pairwise_correlation.set(float(avg_corr))
            result["avg_correlation"] = avg_corr

        cb = snap.get("circuit_breakers", {})
        for venue, is_open in cb.items():
            self._m.circuit_breaker_open.labels(venue=venue).set(1 if is_open else 0)

        amihud = snap.get("amihud", {})
        for sym, val in amihud.items():
            self._m.amihud_illiquidity.labels(symbol=sym).set(float(val))

        result["status"] = "ok"
        return result


class CoordinationScraper(_BaseScraper):
    """
    Polls the Elixir coordination layer at :8781.

    Expected endpoints:
        GET /health          -> {"status": "ok", "node": str}
        GET /metrics/summary -> {"hurst": {sym: float},
                                  "nav_omega": float,
                                  "nav_geodesic_deviation": float,
                                  "param_rollback_count": int,
                                  "active_instruments": [str]}
    """

    name = "coordination"

    def __init__(self, metrics: _PromMetrics, base_url: str = COORD_URL) -> None:
        super().__init__(metrics)
        self._base = base_url.rstrip("/")
        self._session: Optional[Any] = None
        self._prev_rollback_count = 0

    def _sess(self):
        if not _REQUESTS_AVAILABLE:
            return None
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def _get(self, path: str) -> Optional[dict]:
        sess = self._sess()
        if sess is None:
            return None
        try:
            resp = sess.get(f"{self._base}{path}", timeout=HTTP_TIMEOUT_S)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            self._record_error("coordination", "http_error", f"{path}: {exc}")
            return None

    def scrape(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if not _PROM_AVAILABLE:
            return result

        health = self._get("/health")
        if not health or health.get("status") not in ("ok", "healthy"):
            self._m.service_up.labels(service="coordination").set(0)
            result["status"] = "down"
            return result

        self._m.service_up.labels(service="coordination").set(1)

        summary = self._get("/metrics/summary")
        if summary is None:
            summary = {}

        hurst = summary.get("hurst", {})
        for sym, h in hurst.items():
            self._m.hurst_h.labels(symbol=sym).set(float(h))

        omega = summary.get("nav_omega")
        if omega is not None:
            self._m.nav_omega.set(float(omega))
            result["nav_omega"] = omega

        dev = summary.get("nav_geodesic_deviation")
        if dev is not None:
            self._m.nav_geodesic_deviation.set(float(dev))
            result["nav_geodesic_deviation"] = dev

        result["status"] = "ok"
        return result


class SignalEngineScraper(_BaseScraper):
    """
    Reads IPC ring buffer statistics written by the Rust/Go signal engine.

    Expects a JSON status file at SRFM_SIGNAL_STATUS_PATH
    (default: /tmp/srfm_signal_status.json) updated by the engine.

    Schema:
    {
        "bars_processed":   {symbol: {timeframe: int}},
        "signals_generated": {symbol: {direction: int}},
        "bar_latency_ms":   {symbol: [float, ...]},   -- recent sample window
        "signal_ns":        {signal_type: [float, ...]},
        "order_latency_ms": {venue: [float, ...]},
        "updated_at":       "ISO-8601"
    }
    """

    name = "signal_engine"

    STATUS_PATH = os.environ.get(
        "SRFM_SIGNAL_STATUS_PATH", "/tmp/srfm_signal_status.json"
    )

    def __init__(self, metrics: _PromMetrics) -> None:
        super().__init__(metrics)
        self._seen_bars: Dict[Tuple[str, str], int] = {}
        self._seen_signals: Dict[Tuple[str, str], int] = {}

    def scrape(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if not _PROM_AVAILABLE:
            return result

        status_path = Path(self.STATUS_PATH)
        if not status_path.exists():
            self._m.service_up.labels(service="signal_engine").set(0)
            result["status"] = "status_file_missing"
            return result

        try:
            with open(status_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            self._record_error("signal_engine", "parse_error", str(exc))
            self._m.service_up.labels(service="signal_engine").set(0)
            result["status"] = "parse_error"
            return result

        # Staleness check -- file older than 60s means engine is down
        updated_at = data.get("updated_at")
        if updated_at:
            try:
                ts = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - ts).total_seconds()
                if age > 60:
                    self._m.service_up.labels(service="signal_engine").set(0)
                    result["status"] = "stale"
                    return result
            except Exception:
                pass

        self._m.service_up.labels(service="signal_engine").set(1)

        # -- Bars processed counters (delta from last seen) -------------------
        for sym, tf_map in data.get("bars_processed", {}).items():
            for tf, count in tf_map.items():
                key = (sym, tf)
                prev = self._seen_bars.get(key, 0)
                delta = max(0, int(count) - prev)
                if delta > 0:
                    self._m.bars_processed.labels(symbol=sym, timeframe=tf).inc(delta)
                self._seen_bars[key] = int(count)

        # -- Signals generated counters ---------------------------------------
        for sym, dir_map in data.get("signals_generated", {}).items():
            for direction, count in dir_map.items():
                key = (sym, direction)
                prev = self._seen_signals.get(key, 0)
                delta = max(0, int(count) - prev)
                if delta > 0:
                    self._m.signals_generated.labels(
                        symbol=sym, direction=direction
                    ).inc(delta)
                self._seen_signals[key] = int(count)

        # -- Bar processing latency histograms --------------------------------
        for sym, samples in data.get("bar_latency_ms", {}).items():
            for val in samples:
                self._m.bar_processing_latency_ms.labels(symbol=sym).observe(float(val))

        # -- Signal computation histograms ------------------------------------
        for sig_type, samples in data.get("signal_ns", {}).items():
            for val in samples:
                self._m.signal_computation_ns.labels(signal_type=sig_type).observe(float(val))

        # -- Order execution latency ------------------------------------------
        for venue, samples in data.get("order_latency_ms", {}).items():
            for val in samples:
                self._m.order_execution_latency_ms.labels(venue=venue).observe(float(val))

        result["status"] = "ok"
        return result


# ---------------------------------------------------------------------------
# Prometheus HTTP server
# ---------------------------------------------------------------------------

def _make_http_handler(prom: _PromMetrics):
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/metrics", "/metrics/"):
                data = prom.render()
                ct = CONTENT_TYPE_LATEST if _PROM_AVAILABLE else "text/plain; charset=utf-8"
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path in ("/health", "/healthz"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, fmt, *args):  # silence access logs
            pass

    return _Handler


# ---------------------------------------------------------------------------
# MetricsCollector -- main public class
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Orchestrates all scrapers, runs background collection threads,
    and serves /metrics at :9090.

    Usage::

        collector = MetricsCollector()
        collector.start()
        # ... run forever ...
        collector.stop()

    Thread safety: all scraper updates are thread-safe via prometheus_client
    internal locking. The scrapers themselves each run in their own thread
    managed by a ThreadPoolExecutor.
    """

    def __init__(
        self,
        db_path: str = TRADE_DB_PATH,
        risk_api_url: str = RISK_API_URL,
        coord_url: str = COORD_URL,
        port: int = PROMETHEUS_PORT,
        scrape_interval: int = SCRAPE_INTERVAL_S,
    ) -> None:
        self._port = port
        self._interval = scrape_interval
        self._stop_event = threading.Event()
        self._http_server: Optional[HTTPServer] = None

        self._prom = _PromMetrics()

        self._scrapers: List[_BaseScraper] = [
            LiveTraderScraper(self._prom, db_path=db_path),
            RiskAPIScraper(self._prom, base_url=risk_api_url),
            CoordinationScraper(self._prom, base_url=coord_url),
            SignalEngineScraper(self._prom),
        ]

        self._last_results: Dict[str, Any] = {}
        self._results_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=MAX_SCRAPER_WORKERS, thread_name_prefix="srfm_scraper"
        )
        self._collection_thread: Optional[threading.Thread] = None
        self._http_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start HTTP server and background scrape loop."""
        self._stop_event.clear()

        self._http_thread = threading.Thread(
            target=self._run_http,
            daemon=True,
            name="prom_http",
        )
        self._http_thread.start()
        log.info(f"Prometheus metrics server listening on :{self._port}/metrics")

        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,
            name="scrape_loop",
        )
        self._collection_thread.start()
        log.info(f"Scrape loop started -- interval={self._interval}s")

    def stop(self) -> None:
        """Signal background threads to stop and shut down the HTTP server."""
        self._stop_event.set()
        if self._http_server:
            self._http_server.shutdown()
        self._executor.shutdown(wait=False)

    def get_last_results(self) -> Dict[str, Any]:
        """Return a copy of the most recent scrape results."""
        with self._results_lock:
            return dict(self._last_results)

    def force_scrape(self) -> Dict[str, Any]:
        """Run all scrapers synchronously and return results. Used in tests."""
        return self._run_scrape_cycle()

    # Expose prometheus objects for direct mutation by the live trader
    @property
    def metrics(self) -> _PromMetrics:
        return self._prom

    # ------------------------------------------------------------------
    # Internal: collection loop
    # ------------------------------------------------------------------

    def _collection_loop(self) -> None:
        # Stagger first scrape by 2 seconds to let HTTP server start
        time.sleep(2)
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                results = self._run_scrape_cycle()
                with self._results_lock:
                    self._last_results = results
            except Exception as exc:
                log.error(f"Collection loop error: {exc}", exc_info=True)

            elapsed = time.monotonic() - t0
            sleep_for = max(0.0, self._interval - elapsed)
            self._stop_event.wait(timeout=sleep_for)

    def _run_scrape_cycle(self) -> Dict[str, Any]:
        """Fan out to all scrapers in parallel, collect results."""
        futures = {}
        for scraper in self._scrapers:
            fut = self._executor.submit(self._timed_scrape, scraper)
            futures[fut] = scraper.name

        results: Dict[str, Any] = {
            "collected_at": datetime.now(timezone.utc).isoformat()
        }
        for fut, name in futures.items():
            try:
                result = fut.result(timeout=HTTP_TIMEOUT_S + 2)
                results[name] = result
            except Exception as exc:
                log.warning(f"Scraper {name} raised: {exc}")
                results[name] = {"status": "exception", "error": str(exc)}

        return results

    def _timed_scrape(self, scraper: _BaseScraper) -> Dict[str, Any]:
        """Run a single scraper and record its duration."""
        t0 = time.monotonic()
        try:
            return scraper.scrape()
        finally:
            elapsed = time.monotonic() - t0
            if _PROM_AVAILABLE:
                try:
                    self._prom.scrape_duration_s.labels(
                        scraper=scraper.name
                    ).observe(elapsed)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Internal: HTTP server
    # ------------------------------------------------------------------

    def _run_http(self) -> None:
        handler = _make_http_handler(self._prom)
        server = HTTPServer(("0.0.0.0", self._port), handler)
        server.allow_reuse_address = True
        self._http_server = server
        server.serve_forever()


# ---------------------------------------------------------------------------
# Module-level singleton helpers
# ---------------------------------------------------------------------------

_default_collector: Optional[MetricsCollector] = None


def get_collector(**kwargs) -> MetricsCollector:
    """Return the module-level MetricsCollector singleton, creating if needed."""
    global _default_collector
    if _default_collector is None:
        _default_collector = MetricsCollector(**kwargs)
    return _default_collector


# ---------------------------------------------------------------------------
# CLI / standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="SRFM metrics collector")
    parser.add_argument("--port", type=int, default=PROMETHEUS_PORT)
    parser.add_argument("--interval", type=int, default=SCRAPE_INTERVAL_S)
    parser.add_argument("--db", default=TRADE_DB_PATH)
    args = parser.parse_args()

    collector = MetricsCollector(
        db_path=args.db,
        port=args.port,
        scrape_interval=args.interval,
    )
    collector.start()

    log.info("Collector running -- Ctrl-C to stop")
    try:
        while True:
            time.sleep(30)
            results = collector.get_last_results()
            for name, res in results.items():
                if isinstance(res, dict):
                    log.info(f"  {name}: {res.get('status', '?')}")
    except KeyboardInterrupt:
        log.info("Stopping...")
        collector.stop()
