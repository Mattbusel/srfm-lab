"""
metrics_server.py — LARSA observability layer.

Exposes a Prometheus /metrics endpoint (port 9090) and writes to InfluxDB v2
(port 8086) every scrape cycle.

Usage (from live trader):
    from infra.observability.metrics_server import MetricsCollector, start_metrics_server

    collector = MetricsCollector()
    asyncio.ensure_future(start_metrics_server(collector))

    # Every rebalance cycle:
    collector.update(build_state_dict())

State dict schema (all fields optional — missing → last known value):
    {
        "equity":          float,
        "peak_equity":     float,
        "trade_count":     int,

        # per-symbol dicts keyed by LARSA symbol (e.g. "BTC", "SPY"):
        "position_frac":   {sym: float},
        "position_pnl":    {sym: float},   # unrealised P&L $
        "delta_score":     {sym: float},
        "bh_mass":         {sym: {"daily": float, "hourly": float, "m15": float}},
        "bh_active":       {sym: {"daily": bool,  "hourly": bool,  "m15": bool}},
        "tf_score":        {sym: int},      # 0-7
        "atr":             {sym: float},
        "garch_vol":       {sym: float},
        "mean_reversion":  {sym: float},   # OU z-score

        # PID controller state:
        "pid_stale_threshold": float,
        "pid_max_frac":        float,

        # Rolling performance (supplied by TradeLogger):
        "win_rate":        float,
        "rolling_sharpe":  float,
        "rolling_pnl":     float,
        "drawdown":        float,
    }
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

log = logging.getLogger("larsa.metrics")

# ---------------------------------------------------------------------------
# Optional heavy deps — degrade gracefully if not installed
# ---------------------------------------------------------------------------
try:
    from prometheus_client import (
        CollectorRegistry,
        Gauge,
        Counter,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    _PROM_AVAILABLE = True
except ImportError:
    log.warning("prometheus_client not installed — Prometheus endpoint disabled")
    _PROM_AVAILABLE = False

try:
    from influxdb_client import InfluxDBClient, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
    _INFLUX_AVAILABLE = True
except ImportError:
    log.warning("influxdb-client not installed — InfluxDB writes disabled")
    _INFLUX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config (env-overridable)
# ---------------------------------------------------------------------------
PROMETHEUS_PORT  = int(os.environ.get("LARSA_PROM_PORT",    "9090"))
INFLUX_URL       = os.environ.get("LARSA_INFLUX_URL",       "http://localhost:8086")
INFLUX_TOKEN     = os.environ.get("LARSA_INFLUX_TOKEN",     "larsa-super-secret-token")
INFLUX_ORG       = os.environ.get("LARSA_INFLUX_ORG",       "srfm")
INFLUX_BUCKET    = os.environ.get("LARSA_INFLUX_BUCKET",    "larsa_metrics")
INFLUX_WRITE_INTERVAL = int(os.environ.get("LARSA_INFLUX_INTERVAL", "15"))  # seconds

# ---------------------------------------------------------------------------
# Prometheus registry + metrics
# ---------------------------------------------------------------------------

class _PrometheusMetrics:
    """All Prometheus Gauge/Counter objects, lazily created on one registry."""

    def __init__(self):
        if not _PROM_AVAILABLE:
            return
        self.registry = CollectorRegistry()
        R = self.registry

        # ── Account-level ─────────────────────────────────────────────────
        self.equity = Gauge(
            "larsa_equity", "Total account equity in USD", registry=R
        )
        self.trade_count = Counter(
            "larsa_trade_count", "Total trades executed since start", registry=R
        )
        self.win_rate = Gauge(
            "larsa_win_rate", "Rolling 20-trade win rate [0,1]", registry=R
        )
        self.rolling_sharpe = Gauge(
            "larsa_rolling_sharpe", "Rolling 20-trade annualised Sharpe ratio", registry=R
        )
        self.rolling_pnl = Gauge(
            "larsa_rolling_pnl", "Rolling 20-trade total P&L USD", registry=R
        )
        self.drawdown = Gauge(
            "larsa_drawdown", "Current drawdown from peak equity [0,1]", registry=R
        )
        self.pid_stale_threshold = Gauge(
            "larsa_pid_stale_threshold",
            "PID-adjusted stale-move threshold (15m bar fraction)",
            registry=R,
        )
        self.pid_max_frac = Gauge(
            "larsa_pid_max_frac",
            "PID-adjusted delta max fraction cap",
            registry=R,
        )

        # ── Per-symbol ────────────────────────────────────────────────────
        self.position_frac = Gauge(
            "larsa_position_frac",
            "Current position fraction of equity per symbol",
            ["symbol"],
            registry=R,
        )
        self.position_pnl = Gauge(
            "larsa_position_pnl",
            "Unrealised P&L per symbol in USD",
            ["symbol"],
            registry=R,
        )
        self.delta_score = Gauge(
            "larsa_delta_score",
            "Current delta score per symbol (tf × mass × ATR)",
            ["symbol"],
            registry=R,
        )
        self.bh_mass = Gauge(
            "larsa_bh_mass",
            "BH mass per symbol per timeframe",
            ["symbol", "timeframe"],
            registry=R,
        )
        self.bh_active = Gauge(
            "larsa_bh_active",
            "BH active flag (0/1) per symbol per timeframe",
            ["symbol", "timeframe"],
            registry=R,
        )
        self.tf_score = Gauge(
            "larsa_tf_score",
            "Combined timeframe score (0-7) per symbol",
            ["symbol"],
            registry=R,
        )
        self.atr = Gauge(
            "larsa_atr",
            "Current ATR per symbol",
            ["symbol"],
            registry=R,
        )
        self.garch_vol = Gauge(
            "larsa_garch_vol",
            "GARCH volatility forecast per symbol",
            ["symbol"],
            registry=R,
        )
        self.mean_reversion_signal = Gauge(
            "larsa_mean_reversion_signal",
            "OU mean-reversion z-score per symbol",
            ["symbol"],
            registry=R,
        )

    def render(self) -> bytes:
        if not _PROM_AVAILABLE:
            return b""
        return generate_latest(self.registry)


# ---------------------------------------------------------------------------
# InfluxDB writer
# ---------------------------------------------------------------------------

class _InfluxWriter:
    def __init__(self):
        self._client: Optional[Any] = None
        self._write_api: Optional[Any] = None
        self._connected = False
        if _INFLUX_AVAILABLE:
            self._connect()

    def _connect(self):
        try:
            self._client = InfluxDBClient(
                url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG
            )
            self._write_api = self._client.write_api(write_options=SYNCHRONOUS)
            self._connected = True
            log.info(f"InfluxDB connected: {INFLUX_URL} bucket={INFLUX_BUCKET}")
        except Exception as exc:
            log.error(f"InfluxDB connect failed: {exc}")
            self._connected = False

    def write(self, measurement: str, fields: Dict[str, float], tags: Dict[str, str] = None):
        if not self._connected or self._write_api is None:
            return
        try:
            from influxdb_client.client.write_api import SYNCHRONOUS
            from influxdb_client import Point
            p = Point(measurement)
            if tags:
                for k, v in tags.items():
                    p = p.tag(k, v)
            for k, v in fields.items():
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    p = p.field(k, float(v))
            p = p.time(datetime.now(timezone.utc))
            self._write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=p)
        except Exception as exc:
            log.warning(f"InfluxDB write error [{measurement}]: {exc}")
            self._connected = False
            # attempt reconnect on next cycle
            threading.Thread(target=self._connect, daemon=True).start()

    def write_batch(self, points: list):
        """Write a list of influxdb_client.Point objects in one batch."""
        if not self._connected or self._write_api is None:
            return
        try:
            self._write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=points)
        except Exception as exc:
            log.warning(f"InfluxDB batch write error: {exc}")
            self._connected = False
            threading.Thread(target=self._connect, daemon=True).start()

    def close(self):
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Prometheus HTTP handler
# ---------------------------------------------------------------------------

def _make_handler(prom_metrics: _PrometheusMetrics):
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/metrics", "/metrics/"):
                data = prom_metrics.render()
                self.send_response(200)
                self.send_header("Content-Type", CONTENT_TYPE_LATEST if _PROM_AVAILABLE else "text/plain")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path in ("/health", "/healthz"):
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, fmt, *args):
            pass  # silence access logs

    return _Handler


# ---------------------------------------------------------------------------
# MetricsCollector — main public interface
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Thread-safe metrics collector.

    Call `update(state_dict)` from the live trader after every rebalance.
    Runs an HTTP server for Prometheus scraping and a background loop for
    writing to InfluxDB.
    """

    def __init__(self, db_path: str = "data/larsa_trades.db"):
        self._prom = _PrometheusMetrics()
        self._influx = _InfluxWriter()
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {}
        self._prev_trade_count = 0

        # TradeLogger integration
        try:
            from infra.observability.trade_logger import TradeLogger
            self._trade_logger = TradeLogger(db_path)
        except ImportError:
            try:
                from trade_logger import TradeLogger
                self._trade_logger = TradeLogger(db_path)
            except ImportError:
                self._trade_logger = None
                log.warning("TradeLogger not available — performance metrics disabled")

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def update(self, state: Dict[str, Any]):
        """
        Ingest a state snapshot from the live trader.
        Thread-safe. Call this after every rebalance.
        """
        with self._lock:
            self._state.update(state)
        self._apply_to_prometheus(state)
        # InfluxDB write happens in background loop

    def record_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        pnl: float,
        entry_price: Optional[float] = None,
        bars_held: int = 0,
        equity_after: Optional[float] = None,
        trade_duration_s: Optional[float] = None,
    ):
        """Convenience: log a trade to SQLite and increment Prometheus counter."""
        if self._trade_logger:
            self._trade_logger.log_trade(
                symbol, side, qty, price, pnl,
                entry_price=entry_price,
                bars_held=bars_held,
                equity_after=equity_after,
                trade_duration_s=trade_duration_s,
            )
        if _PROM_AVAILABLE:
            self._prom.trade_count.inc()

    def record_equity_snapshot(self, equity: float, positions: Dict[str, float]):
        if self._trade_logger:
            self._trade_logger.log_equity_snapshot(equity, positions)

    # ------------------------------------------------------------------ #
    # Internal: Prometheus updates                                          #
    # ------------------------------------------------------------------ #

    def _apply_to_prometheus(self, state: Dict[str, Any]):
        if not _PROM_AVAILABLE:
            return
        p = self._prom

        if "equity" in state:
            p.equity.set(state["equity"])
        if "win_rate" in state:
            p.win_rate.set(state["win_rate"])
        if "rolling_sharpe" in state:
            p.rolling_sharpe.set(state["rolling_sharpe"])
        if "rolling_pnl" in state:
            p.rolling_pnl.set(state["rolling_pnl"])
        if "drawdown" in state:
            p.drawdown.set(state["drawdown"])
        if "pid_stale_threshold" in state:
            p.pid_stale_threshold.set(state["pid_stale_threshold"])
        if "pid_max_frac" in state:
            p.pid_max_frac.set(state["pid_max_frac"])

        for sym, val in state.get("position_frac", {}).items():
            p.position_frac.labels(symbol=sym).set(val)
        for sym, val in state.get("position_pnl", {}).items():
            p.position_pnl.labels(symbol=sym).set(val)
        for sym, val in state.get("delta_score", {}).items():
            p.delta_score.labels(symbol=sym).set(val)
        for sym, val in state.get("tf_score", {}).items():
            p.tf_score.labels(symbol=sym).set(val)
        for sym, val in state.get("atr", {}).items():
            p.atr.labels(symbol=sym).set(val)
        for sym, val in state.get("garch_vol", {}).items():
            p.garch_vol.labels(symbol=sym).set(val)
        for sym, val in state.get("mean_reversion", {}).items():
            p.mean_reversion_signal.labels(symbol=sym).set(val)

        bh_mass   = state.get("bh_mass", {})
        bh_active = state.get("bh_active", {})
        tf_map = {"daily": "daily", "hourly": "hourly", "m15": "15m"}
        for sym, masses in bh_mass.items():
            for tf_key, tf_label in tf_map.items():
                val = masses.get(tf_key, 0.0)
                p.bh_mass.labels(symbol=sym, timeframe=tf_label).set(val)
        for sym, flags in bh_active.items():
            for tf_key, tf_label in tf_map.items():
                val = 1 if flags.get(tf_key) else 0
                p.bh_active.labels(symbol=sym, timeframe=tf_label).set(val)

    # ------------------------------------------------------------------ #
    # Internal: InfluxDB batch write                                        #
    # ------------------------------------------------------------------ #

    def _write_to_influx(self):
        """Snapshot current state and write all metrics to InfluxDB."""
        if not _INFLUX_AVAILABLE:
            return

        with self._lock:
            state = dict(self._state)

        try:
            from influxdb_client import Point
        except ImportError:
            return

        now = datetime.now(timezone.utc)
        points: List[Any] = []

        # Account metrics
        account_fields: Dict[str, float] = {}
        for field in ["equity", "win_rate", "rolling_sharpe", "rolling_pnl",
                      "drawdown", "pid_stale_threshold", "pid_max_frac", "trade_count"]:
            if field in state:
                account_fields[field] = float(state[field])

        # Augment with TradeLogger rolling stats
        if self._trade_logger:
            try:
                stats = self._trade_logger.get_rolling_stats(20)
                account_fields.setdefault("win_rate",       stats["win_rate"])
                account_fields.setdefault("rolling_sharpe", stats["sharpe"])
                account_fields.setdefault("rolling_pnl",    stats["total_pnl"])
                account_fields.setdefault("drawdown",       self._trade_logger.get_current_drawdown())
                account_fields.setdefault("trade_count",    float(stats["trade_count"]))
            except Exception as exc:
                log.debug(f"TradeLogger stats: {exc}")

        if account_fields:
            p = Point("larsa_account")
            for k, v in account_fields.items():
                p = p.field(k, v)
            p = p.time(now)
            points.append(p)

        # Per-symbol metrics
        symbols = set()
        for key in ["position_frac", "position_pnl", "delta_score", "tf_score",
                    "atr", "garch_vol", "mean_reversion"]:
            symbols.update(state.get(key, {}).keys())
        symbols.update(state.get("bh_mass", {}).keys())

        for sym in symbols:
            p = Point("larsa_symbol").tag("symbol", sym)
            wrote = False

            for field, key in [
                ("position_frac",  "position_frac"),
                ("position_pnl",   "position_pnl"),
                ("delta_score",    "delta_score"),
                ("tf_score",       "tf_score"),
                ("atr",            "atr"),
                ("garch_vol",      "garch_vol"),
                ("ou_zscore",      "mean_reversion"),
            ]:
                val = state.get(key, {}).get(sym)
                if val is not None:
                    p = p.field(field, float(val))
                    wrote = True

            masses = state.get("bh_mass", {}).get(sym, {})
            flags  = state.get("bh_active", {}).get(sym, {})
            for tf_key, tf_label in [("daily","daily"), ("hourly","hourly"), ("m15","15m")]:
                mass = masses.get(tf_key)
                flag = flags.get(tf_key)
                if mass is not None:
                    p = p.field(f"bh_mass_{tf_label}", float(mass))
                    wrote = True
                if flag is not None:
                    p = p.field(f"bh_active_{tf_label}", int(flag))
                    wrote = True

            if wrote:
                p = p.time(now)
                points.append(p)

        if points:
            self._influx.write_batch(points)
            log.debug(f"Wrote {len(points)} points to InfluxDB")

    # ------------------------------------------------------------------ #
    # Background loops                                                      #
    # ------------------------------------------------------------------ #

    async def _influx_loop(self):
        """Background asyncio task: write to InfluxDB every N seconds."""
        while True:
            await asyncio.sleep(INFLUX_WRITE_INTERVAL)
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._write_to_influx)
            except Exception as exc:
                log.error(f"InfluxDB loop error: {exc}")

    def _run_http_server(self):
        """Run the Prometheus HTTP server in a daemon thread."""
        handler = _make_handler(self._prom)
        server = HTTPServer(("0.0.0.0", PROMETHEUS_PORT), handler)
        log.info(f"Prometheus metrics server on :{PROMETHEUS_PORT}/metrics")
        server.serve_forever()

    def close(self):
        self._influx.close()
        if self._trade_logger:
            self._trade_logger.close()


# ---------------------------------------------------------------------------
# Entry point: start background tasks
# ---------------------------------------------------------------------------

async def start_metrics_server(collector: Optional[MetricsCollector] = None) -> MetricsCollector:
    """
    Start the observability stack as asyncio background tasks.

    Returns the MetricsCollector for use by the caller.

    Example (from live trader):
        collector = await start_metrics_server()
        # later:
        collector.update(state_dict)
    """
    if collector is None:
        collector = MetricsCollector()

    # Prometheus HTTP server in daemon thread
    t = threading.Thread(target=collector._run_http_server, daemon=True, name="prom_http")
    t.start()

    # InfluxDB write loop as asyncio task
    asyncio.ensure_future(collector._influx_loop())

    log.info(
        f"Observability stack started — "
        f"Prometheus :{PROMETHEUS_PORT}/metrics  "
        f"InfluxDB {INFLUX_URL}/{INFLUX_BUCKET}"
    )
    return collector


# ---------------------------------------------------------------------------
# Standalone mode: generate synthetic metrics for demo/testing
# ---------------------------------------------------------------------------

async def _demo_loop(collector: MetricsCollector):
    """Feed the collector with synthetic data to test the stack end-to-end."""
    import random

    SYMS = ["BTC", "ETH", "SOL", "XRP", "SPY", "QQQ", "DIA"]
    equity = 100_000.0
    peak   = equity
    trade_n = 0

    while True:
        equity += random.gauss(0, 200)
        peak = max(peak, equity)
        trade_n += random.randint(0, 2)

        state = {
            "equity":    equity,
            "peak_equity": peak,
            "trade_count": trade_n,
            "win_rate":    random.uniform(0.45, 0.65),
            "rolling_sharpe": random.gauss(0.8, 0.4),
            "rolling_pnl": random.gauss(500, 1000),
            "drawdown": (peak - equity) / (peak + 1e-9),
            "pid_stale_threshold": random.uniform(0.0008, 0.002),
            "pid_max_frac": random.uniform(0.50, 0.85),
            "position_frac": {},
            "position_pnl":  {},
            "delta_score":   {},
            "bh_mass":       {},
            "bh_active":     {},
            "tf_score":      {},
            "atr":           {},
            "garch_vol":     {},
            "mean_reversion": {},
        }

        for sym in SYMS:
            frac = random.uniform(-0.3, 0.4)
            state["position_frac"][sym]  = frac
            state["position_pnl"][sym]   = frac * equity * random.gauss(0.01, 0.05)
            state["delta_score"][sym]    = max(0, random.gauss(0.5, 0.3))
            d_mass = random.uniform(0.5, 2.2)
            h_mass = random.uniform(0.3, 2.0)
            m_mass = random.uniform(0.2, 1.8)
            state["bh_mass"][sym]   = {"daily": d_mass, "hourly": h_mass, "m15": m_mass}
            d_act = d_mass > 1.5
            h_act = h_mass > 1.5
            m_act = m_mass > 1.5
            state["bh_active"][sym] = {"daily": d_act, "hourly": h_act, "m15": m_act}
            state["tf_score"][sym]  = (4 if d_act else 0) + (2 if h_act else 0) + (1 if m_act else 0)
            state["atr"][sym]       = random.uniform(10, 500)
            state["garch_vol"][sym] = random.uniform(0.01, 0.08)
            state["mean_reversion"][sym] = random.gauss(0, 1.5)

        collector.update(state)
        log.info(f"[demo] equity={equity:,.0f}  drawdown={state['drawdown']:.3%}")
        await asyncio.sleep(15)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )
    collector = await start_metrics_server()
    await _demo_loop(collector)


if __name__ == "__main__":
    asyncio.run(main())
