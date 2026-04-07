"""
dashboard_api.py -- FastAPI dashboard service for SRFM trading system metrics.

Runs at :9091. Serves current metrics, historical data, and alert status
to the TypeScript frontend dashboards.

Endpoints:
    GET  /metrics/live                -- current snapshot of all gauges
    GET  /metrics/history             -- historical metric values from SQLite
    GET  /alerts/active               -- currently firing alerts
    GET  /alerts/history              -- past alerts
    GET  /system/health               -- aggregate health of all components
    WS   /ws/metrics                  -- WebSocket: push metric updates every 5s

Usage:
    uvicorn infra.observability.dashboard_api:app --port 9091
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiosqlite

log = logging.getLogger("srfm.dashboard_api")

# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------
try:
    import httpx as _httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

try:
    from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    log.error("fastapi not installed -- dashboard API cannot start")

try:
    from pydantic import BaseModel
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DASHBOARD_PORT   = int(os.environ.get("SRFM_DASHBOARD_PORT",  "9091"))
AUDIT_DB_PATH    = os.environ.get("SRFM_AUDIT_DB",            "data/audit.db")
TRADE_DB_PATH    = os.environ.get("SRFM_TRADE_DB",            "data/larsa_trades.db")
RISK_API_URL     = os.environ.get("SRFM_RISK_API_URL",        "http://localhost:8791")
COORD_URL        = os.environ.get("SRFM_COORD_URL",           "http://localhost:8781")
PROM_URL         = os.environ.get("SRFM_PROM_URL",            "http://localhost:9090")
WS_PUSH_INTERVAL = float(os.environ.get("SRFM_WS_INTERVAL",  "5"))
HTTP_TIMEOUT     = float(os.environ.get("SRFM_HTTP_TIMEOUT",  "5"))

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]


# ---------------------------------------------------------------------------
# State cache -- updated by background task
# ---------------------------------------------------------------------------

class _MetricsCache:
    """Thread-safe in-memory cache for the latest system metrics snapshot."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._snapshot: Dict[str, Any] = {}
        self._updated_at: Optional[datetime] = None

    async def update(self, data: Dict[str, Any]) -> None:
        async with self._lock:
            self._snapshot = data
            self._updated_at = datetime.now(timezone.utc)

    async def get(self) -> Dict[str, Any]:
        async with self._lock:
            return {
                **self._snapshot,
                "cache_updated_at": self._updated_at.isoformat() if self._updated_at else None,
            }

    async def is_stale(self, max_age_s: float = 60.0) -> bool:
        async with self._lock:
            if self._updated_at is None:
                return True
            age = (datetime.now(timezone.utc) - self._updated_at).total_seconds()
            return age > max_age_s


_cache = _MetricsCache()

# WebSocket connection registry
_ws_clients: List[WebSocket] = []
_ws_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

async def _query_audit_db(
    sql: str,
    params: tuple = (),
    db_path: str = AUDIT_DB_PATH,
) -> List[Dict[str, Any]]:
    """Run a read query against the audit SQLite DB."""
    path = Path(db_path)
    if not path.exists():
        return []
    async with aiosqlite.connect(str(path)) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def _query_trade_db(
    sql: str,
    params: tuple = (),
    db_path: str = TRADE_DB_PATH,
) -> List[Dict[str, Any]]:
    """Run a read query against the trade SQLite DB."""
    path = Path(db_path)
    if not path.exists():
        return []
    async with aiosqlite.connect(str(path)) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Metrics aggregator
# ---------------------------------------------------------------------------

async def _fetch_component_health() -> Dict[str, Any]:
    """Poll health endpoints of all microservices."""
    services = {
        "risk_api":    f"{RISK_API_URL}/health",
        "coordination": f"{COORD_URL}/health",
        "prometheus":  f"{PROM_URL}/health",
    }
    health: Dict[str, Any] = {}

    if not _HTTPX_AVAILABLE:
        return {svc: {"status": "unknown", "reason": "httpx not installed"}
                for svc in services}

    async with _httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        for svc, url in services.items():
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    health[svc] = {"status": "ok"}
                else:
                    health[svc] = {"status": "degraded", "http_code": resp.status_code}
            except Exception as exc:
                health[svc] = {"status": "down", "reason": str(exc)}

    # Check live_trader via DB existence
    health["live_trader"] = {
        "status": "ok" if Path(TRADE_DB_PATH).exists() else "down"
    }

    return health


async def _fetch_risk_snapshot() -> Dict[str, Any]:
    """Fetch current risk metrics from the risk API."""
    if not _HTTPX_AVAILABLE:
        return {}
    try:
        async with _httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.get(f"{RISK_API_URL}/risk/snapshot")
            if resp.status_code == 200:
                return resp.json()
    except Exception as exc:
        log.debug(f"Risk snapshot fetch failed: {exc}")
    return {}


async def _fetch_coord_summary() -> Dict[str, Any]:
    """Fetch metrics summary from Elixir coordinator."""
    if not _HTTPX_AVAILABLE:
        return {}
    try:
        async with _httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await client.get(f"{COORD_URL}/metrics/summary")
            if resp.status_code == 200:
                return resp.json()
    except Exception as exc:
        log.debug(f"Coord summary fetch failed: {exc}")
    return {}


async def _build_live_snapshot() -> Dict[str, Any]:
    """Aggregate all live metrics into one dict."""
    # Run fetches concurrently
    results = await asyncio.gather(
        _fetch_component_health(),
        _fetch_risk_snapshot(),
        _fetch_coord_summary(),
        return_exceptions=True,
    )

    component_health = results[0] if not isinstance(results[0], Exception) else {}
    risk_snap        = results[1] if not isinstance(results[1], Exception) else {}
    coord_summary    = results[2] if not isinstance(results[2], Exception) else {}

    # Latest equity from trade DB
    equity_row = await _query_trade_db(
        "SELECT equity FROM equity_snapshots ORDER BY ts DESC LIMIT 1"
    )
    equity = equity_row[0]["equity"] if equity_row else None

    # Current drawdown
    dd_rows = await _query_trade_db(
        "SELECT equity FROM equity_snapshots ORDER BY ts DESC LIMIT 100"
    )
    drawdown = None
    if dd_rows:
        equities = [r["equity"] for r in reversed(dd_rows)]
        peak = max(equities)
        if peak > 0:
            drawdown = (peak - equities[-1]) / peak

    # Latest regime per symbol
    regime_rows = await _query_trade_db(
        """
        SELECT symbol, d_bh_mass, h_bh_mass, m15_bh_mass, garch_vol, MAX(ts) AS ts
        FROM regime_log GROUP BY symbol
        """
    )
    bh_mass: Dict[str, Any] = {}
    garch_vol: Dict[str, float] = {}
    for row in regime_rows:
        sym = row["symbol"]
        bh_mass[sym] = {
            "daily": row["d_bh_mass"],
            "hourly": row["h_bh_mass"],
            "15m": row["m15_bh_mass"],
        }
        if row["garch_vol"] is not None:
            garch_vol[sym] = row["garch_vol"]

    # Positions
    pos_rows = await _query_trade_db(
        "SELECT sym, qty, current_price, unrealized_pnl FROM positions"
    )
    positions: Dict[str, Any] = {}
    for row in pos_rows:
        positions[row["sym"]] = {
            "qty": row["qty"],
            "current_price": row["current_price"],
            "unrealized_pnl": row["unrealized_pnl"],
        }

    return {
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "equity":            equity,
        "drawdown":          drawdown,
        "bh_mass":           bh_mass,
        "garch_vol":         garch_vol,
        "positions":         positions,
        "service_health":    component_health,
        "risk":              risk_snap,
        "coordination":      coord_summary,
        "hurst":             coord_summary.get("hurst", {}),
        "nav_omega":         coord_summary.get("nav_omega"),
        "nav_geodesic_deviation": coord_summary.get("nav_geodesic_deviation"),
        "var_95":            risk_snap.get("var_95"),
        "avg_correlation":   risk_snap.get("avg_correlation"),
        "circuit_breakers":  risk_snap.get("circuit_breakers", {}),
        "amihud":            risk_snap.get("amihud", {}),
    }


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

async def _metrics_refresh_loop() -> None:
    """Refresh the metrics cache every WS_PUSH_INTERVAL seconds."""
    while True:
        try:
            snap = await _build_live_snapshot()
            await _cache.update(snap)
        except Exception as exc:
            log.error(f"Metrics refresh error: {exc}", exc_info=True)
        await asyncio.sleep(WS_PUSH_INTERVAL)


async def _ws_broadcast_loop() -> None:
    """Broadcast latest metrics to all connected WebSocket clients."""
    while True:
        await asyncio.sleep(WS_PUSH_INTERVAL)
        if not _ws_clients:
            continue
        snapshot = await _cache.get()
        payload = json.dumps(snapshot)
        disconnected: List[WebSocket] = []

        async with _ws_lock:
            for ws in list(_ws_clients):
                try:
                    await ws.send_text(payload)
                except Exception:
                    disconnected.append(ws)

        if disconnected:
            async with _ws_lock:
                for ws in disconnected:
                    if ws in _ws_clients:
                        _ws_clients.remove(ws)
            log.debug(f"Removed {len(disconnected)} disconnected WS clients")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

if not _FASTAPI_AVAILABLE:
    raise RuntimeError("fastapi is required for dashboard_api")


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Start background tasks
    refresh_task  = asyncio.create_task(_metrics_refresh_loop())
    broadcast_task = asyncio.create_task(_ws_broadcast_loop())
    log.info(f"Dashboard API starting -- port {DASHBOARD_PORT}")
    yield
    refresh_task.cancel()
    broadcast_task.cancel()
    log.info("Dashboard API shut down")


app = FastAPI(
    title="SRFM Dashboard API",
    description="Live metrics and alert status for SRFM quantitative trading system",
    version="1.0.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/metrics/live",
    summary="Current snapshot of all system gauges",
    tags=["metrics"],
)
async def get_live_metrics() -> JSONResponse:
    """
    Returns the latest cached snapshot of all system metrics including
    equity, positions, BH mass, GARCH vol, Hurst exponents, and risk metrics.
    """
    snapshot = await _cache.get()
    stale = await _cache.is_stale(max_age_s=30.0)
    return JSONResponse(
        content={
            "status": "stale" if stale else "ok",
            "data": snapshot,
        }
    )


@app.get(
    "/metrics/history",
    summary="Historical metric values from SQLite",
    tags=["metrics"],
)
async def get_metrics_history(
    metric: str = Query(..., description="Metric name: bh_mass, equity, garch_vol, regime"),
    symbol: Optional[str] = Query(None, description="Filter by symbol (e.g. BTC)"),
    since: Optional[str] = Query(
        None,
        description="ISO-8601 datetime -- default 7 days ago",
    ),
    limit: int = Query(500, ge=1, le=5000),
) -> JSONResponse:
    """
    Returns historical values for a given metric from the SQLite trade database.
    Supported metrics: equity, bh_mass, garch_vol, regime, trades.
    """
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(400, f"Invalid since datetime: {since!r}")
    else:
        since_dt = datetime.now(timezone.utc) - timedelta(days=7)

    since_str = since_dt.isoformat()

    if metric == "equity":
        rows = await _query_trade_db(
            "SELECT ts, equity FROM equity_snapshots WHERE ts >= ? ORDER BY ts LIMIT ?",
            (since_str, limit),
        )
        return JSONResponse({"metric": "equity", "data": rows})

    elif metric == "bh_mass":
        sql = """
            SELECT ts, symbol, d_bh_mass, h_bh_mass, m15_bh_mass
            FROM regime_log
            WHERE ts >= ?
        """
        params: list = [since_str]
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        sql += " ORDER BY ts LIMIT ?"
        params.append(limit)
        rows = await _query_trade_db(sql, tuple(params))
        return JSONResponse({"metric": "bh_mass", "symbol": symbol, "data": rows})

    elif metric == "garch_vol":
        sql = """
            SELECT ts, symbol, garch_vol
            FROM regime_log
            WHERE ts >= ? AND garch_vol IS NOT NULL
        """
        params = [since_str]
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        sql += " ORDER BY ts LIMIT ?"
        params.append(limit)
        rows = await _query_trade_db(sql, tuple(params))
        return JSONResponse({"metric": "garch_vol", "symbol": symbol, "data": rows})

    elif metric == "regime":
        sql = """
            SELECT ts, symbol, d_bh_mass, h_bh_mass, m15_bh_mass,
                   d_bh_active, h_bh_active, m15_bh_active,
                   tf_score, delta_score, atr, garch_vol, ou_zscore
            FROM regime_log
            WHERE ts >= ?
        """
        params = [since_str]
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        sql += " ORDER BY ts LIMIT ?"
        params.append(limit)
        rows = await _query_trade_db(sql, tuple(params))
        return JSONResponse({"metric": "regime", "symbol": symbol, "data": rows})

    elif metric == "trades":
        sql = """
            SELECT ts, symbol, side, qty, price, pnl
            FROM trades
            WHERE ts >= ?
        """
        params = [since_str]
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        sql += " ORDER BY ts LIMIT ?"
        params.append(limit)
        rows = await _query_trade_db(sql, tuple(params))
        return JSONResponse({"metric": "trades", "symbol": symbol, "data": rows})

    else:
        raise HTTPException(
            400,
            f"Unknown metric {metric!r}. "
            "Supported: equity, bh_mass, garch_vol, regime, trades",
        )


@app.get(
    "/alerts/active",
    summary="Currently firing alerts",
    tags=["alerts"],
)
async def get_active_alerts() -> JSONResponse:
    """Returns all alerts that have not yet been resolved."""
    rows = await _query_audit_db(
        "SELECT * FROM alerts WHERE resolved_at IS NULL ORDER BY timestamp DESC"
    )
    return JSONResponse({"count": len(rows), "alerts": rows})


@app.get(
    "/alerts/history",
    summary="Historical alerts",
    tags=["alerts"],
)
async def get_alert_history(
    since: Optional[str] = Query(None, description="ISO-8601 datetime"),
    severity: Optional[str] = Query(
        None, description="Filter by severity: INFO, WARNING, CRITICAL"
    ),
    rule_name: Optional[str] = Query(None, description="Filter by rule name"),
    limit: int = Query(200, ge=1, le=2000),
) -> JSONResponse:
    """Returns historical alert records from the audit database."""
    conditions = []
    params: list = []

    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(400, f"Invalid since: {since!r}")
        conditions.append("timestamp >= ?")
        params.append(since_dt.isoformat())

    if severity:
        sev_upper = severity.upper()
        if sev_upper not in ("INFO", "WARNING", "CRITICAL"):
            raise HTTPException(400, f"Invalid severity: {severity!r}")
        conditions.append("severity = ?")
        params.append(sev_upper)

    if rule_name:
        conditions.append("rule_name = ?")
        params.append(rule_name)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)

    rows = await _query_audit_db(
        f"SELECT * FROM alerts {where} ORDER BY timestamp DESC LIMIT ?",
        tuple(params),
    )
    return JSONResponse({"count": len(rows), "alerts": rows})


@app.get(
    "/system/health",
    summary="Aggregate health of all system components",
    tags=["system"],
)
async def get_system_health() -> JSONResponse:
    """
    Polls health endpoints of all microservices and returns aggregate status.
    Overall status is OK only if all critical services are healthy.
    """
    component_health = await _fetch_component_health()

    critical_services = {"risk_api", "live_trader"}
    all_ok = all(
        component_health.get(svc, {}).get("status") == "ok"
        for svc in critical_services
    )

    return JSONResponse(
        content={
            "status":     "ok" if all_ok else "degraded",
            "components": component_health,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        },
        status_code=200 if all_ok else 503,
    )


@app.websocket("/ws/metrics")
async def ws_metrics(websocket: WebSocket) -> None:
    """
    WebSocket endpoint -- pushes metric updates every WS_PUSH_INTERVAL seconds.

    The client receives a JSON object matching the /metrics/live response shape.
    """
    await websocket.accept()

    async with _ws_lock:
        _ws_clients.append(websocket)

    log.debug(
        f"WebSocket client connected -- total={len(_ws_clients)}"
    )

    # Send current snapshot immediately on connect
    try:
        snap = await _cache.get()
        await websocket.send_text(json.dumps(snap))
    except Exception:
        pass

    try:
        # Keep connection alive until client disconnects
        while True:
            try:
                # ping-pong to detect disconnect
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=WS_PUSH_INTERVAL * 3
                )
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Timeout means no client message -- that is OK, broadcast loop handles pushes
                pass
            except WebSocketDisconnect:
                break
    except Exception as exc:
        log.debug(f"WS client error: {exc}")
    finally:
        async with _ws_lock:
            if websocket in _ws_clients:
                _ws_clients.remove(websocket)
        log.debug(f"WebSocket client disconnected -- total={len(_ws_clients)}")


@app.get("/health", include_in_schema=False)
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    )
    uvicorn.run(
        "infra.observability.dashboard_api:app",
        host="0.0.0.0",
        port=DASHBOARD_PORT,
        reload=False,
        log_level="info",
    )
