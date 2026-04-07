"""
execution/risk/risk_api.py
==========================
FastAPI risk monitoring service for the SRFM Lab trading system.

Endpoints
---------
GET /risk/portfolio     -- current portfolio VaR (all three methods), concentration, P&L
GET /risk/limits        -- all limits with current values and breach status
GET /risk/attribution   -- factor attribution for last N days (?days=7)
GET /risk/correlation   -- current correlation matrix as JSON
GET /risk/health        -- service health check

Runs on port 8791.
All database reads use WAL mode for safe concurrent access.

Run standalone:
    python -m execution.risk.risk_api
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from execution.risk.live_var import (
    VaRMonitor,
    PortfolioSnapshot,
    PositionSnapshot,
    snapshot_from_db,
)
from execution.risk.attribution import AttributionReport, read_live_trades
from execution.risk.limits import (
    RiskLimitConfig,
    LimitChecker,
    DrawdownGuard,
)
from execution.risk.correlation_monitor import CorrelationMonitor

log = logging.getLogger("execution.risk.api")

_DB_PATH = Path(__file__).parents[2] / "execution" / "live_trades.db"
_PORT = 8791


# ---------------------------------------------------------------------------
# Application state (initialised at startup)
# ---------------------------------------------------------------------------

class _AppState:
    """Holds shared service state."""
    db_path: Path = _DB_PATH
    var_monitor: Optional[VaRMonitor] = None
    corr_monitor: Optional[CorrelationMonitor] = None
    limit_config: Optional[RiskLimitConfig] = None
    limit_checker: Optional[LimitChecker] = None
    attribution_report: Optional[AttributionReport] = None
    drawdown_guard: Optional[DrawdownGuard] = None
    started_at: Optional[datetime] = None
    equity: float = 100_000.0
    initial_equity: float = 100_000.0


_state = _AppState()


# ---------------------------------------------------------------------------
# Database helpers (WAL-safe reads)
# ---------------------------------------------------------------------------

def _db_connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _read_latest_var(db_path: Path, n: int = 4) -> List[Dict]:
    try:
        with _db_connect(db_path) as conn:
            rows = conn.execute(
                """SELECT method, var_95, var_99, cvar_95, cvar_99,
                          consensus_var99, equity, n_positions, breach_flag, timestamp
                   FROM risk_metrics ORDER BY id DESC LIMIT ?""",
                (n,),
            ).fetchall()
    except Exception as exc:
        log.error("_read_latest_var: %s", exc)
        return []
    cols = ["method", "var_95", "var_99", "cvar_95", "cvar_99",
            "consensus_var99", "equity", "n_positions", "breach_flag", "timestamp"]
    return [dict(zip(cols, row)) for row in rows]


def _read_daily_pnl(db_path: Path) -> float:
    """Approximate today's P&L from trade_pnl rows with today's exit_time."""
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with _db_connect(db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0.0) FROM trade_pnl WHERE exit_time LIKE ?",
                (f"{today}%",),
            ).fetchone()
        return float(row[0]) if row else 0.0
    except Exception as exc:
        log.error("_read_daily_pnl: %s", exc)
        return 0.0


def _compute_portfolio_metrics(db_path: Path, equity: float) -> Dict:
    """
    Derive gross exposure, max position fraction, and per-symbol notionals
    from open live_trades positions.
    """
    try:
        with _db_connect(db_path) as conn:
            df = pd.read_sql_query(
                "SELECT symbol, side, qty, price FROM live_trades ORDER BY fill_time",
                conn,
            )
    except Exception as exc:
        log.error("_compute_portfolio_metrics: %s", exc)
        return {
            "gross_exposure_frac": 0.0,
            "max_position_frac": 0.0,
            "per_symbol_notionals": {},
        }

    if df.empty:
        return {
            "gross_exposure_frac": 0.0,
            "max_position_frac": 0.0,
            "per_symbol_notionals": {},
        }

    per_sym: Dict[str, float] = {}
    for sym, grp in df.groupby("symbol"):
        net = 0.0
        last_price = 0.0
        for _, row in grp.iterrows():
            sign = 1.0 if row["side"] in ("buy", "long") else -1.0
            net += sign * row["qty"]
            last_price = row["price"]
        notional = abs(net * last_price)
        if notional > 1e-6:
            per_sym[str(sym)] = notional

    total_gross = sum(per_sym.values())
    gross_frac = total_gross / max(equity, 1.0)
    max_frac = max((v / max(equity, 1.0) for v in per_sym.values()), default=0.0)
    return {
        "gross_exposure_frac": gross_frac,
        "max_position_frac": max_frac,
        "per_symbol_notionals": per_sym,
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:
    app = FastAPI(
        title="SRFM Risk API",
        description="Real-time risk monitoring for the SRFM Lab trading system.",
        version="1.0.0",
    )

    @app.on_event("startup")
    async def startup_event() -> None:
        _state.started_at = datetime.now(timezone.utc)
        _state.limit_config = RiskLimitConfig()
        _state.limit_checker = LimitChecker(config=_state.limit_config)
        _state.var_monitor = VaRMonitor(db_path=_state.db_path)
        _state.attribution_report = AttributionReport(db_path=_state.db_path)
        # Determine symbols from live_trades
        try:
            with _db_connect(_state.db_path) as conn:
                syms = [
                    r[0] for r in
                    conn.execute("SELECT DISTINCT symbol FROM live_trades").fetchall()
                ]
        except Exception:
            syms = []
        _state.corr_monitor = CorrelationMonitor(
            symbols=syms or ["BTC", "ETH", "SPY"],
            db_path=_state.db_path,
        )
        _state.drawdown_guard = DrawdownGuard(initial_equity=_state.initial_equity)
        log.info("Risk API started on port %d", _PORT)

    # ------------------------------------------------------------------ #
    # GET /risk/health                                                     #
    # ------------------------------------------------------------------ #

    @app.get("/risk/health")
    async def health() -> Dict:
        uptime_secs = 0.0
        if _state.started_at:
            uptime_secs = (datetime.now(timezone.utc) - _state.started_at).total_seconds()
        try:
            with _db_connect(_state.db_path) as conn:
                conn.execute("SELECT 1")
            db_ok = True
        except Exception:
            db_ok = False
        return {
            "status": "ok" if db_ok else "degraded",
            "db_connected": db_ok,
            "uptime_seconds": round(uptime_secs, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------ #
    # GET /risk/portfolio                                                  #
    # ------------------------------------------------------------------ #

    @app.get("/risk/portfolio")
    async def portfolio_risk() -> Dict:
        """
        Returns current portfolio VaR from all three methods, concentration
        metrics, and current-day P&L.
        """
        equity = _state.equity
        daily_pnl = _read_daily_pnl(_state.db_path)
        metrics = _compute_portfolio_metrics(_state.db_path, equity)
        var_rows = _read_latest_var(_state.db_path, n=4)

        # Organise VaR rows by method
        var_by_method: Dict[str, Any] = {}
        for row in var_rows:
            m = row.get("method", "unknown")
            var_by_method[m] = {
                "var_95": round(float(row["var_95"] or 0), 2),
                "var_99": round(float(row["var_99"] or 0), 2),
                "cvar_95": round(float(row["cvar_95"] or 0), 2),
                "cvar_99": round(float(row["cvar_99"] or 0), 2),
                "breach_flag": bool(row["breach_flag"]),
                "timestamp": row["timestamp"],
            }

        snap = snapshot_from_db(db_path=_state.db_path, equity=equity)

        dd_status = {}
        if _state.drawdown_guard:
            _state.drawdown_guard.update(equity)
            dd_status = _state.drawdown_guard.status()

        return {
            "equity": equity,
            "daily_pnl": round(daily_pnl, 2),
            "n_positions": len(snap.positions),
            "gross_exposure_frac": round(metrics["gross_exposure_frac"], 4),
            "max_position_frac": round(metrics["max_position_frac"], 4),
            "var": var_by_method,
            "drawdown": dd_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------ #
    # GET /risk/limits                                                     #
    # ------------------------------------------------------------------ #

    @app.get("/risk/limits")
    async def risk_limits() -> Dict:
        """
        Returns all configured risk limits with their current measured values
        and breach status.
        """
        if _state.limit_checker is None or _state.limit_config is None:
            raise HTTPException(status_code=503, detail="limit checker not initialised")

        equity = _state.equity
        daily_pnl = _read_daily_pnl(_state.db_path)
        metrics = _compute_portfolio_metrics(_state.db_path, equity)

        # Get latest VaR99 from DB
        var_rows = _read_latest_var(_state.db_path, n=1)
        var99_frac = 0.0
        if var_rows:
            cv = var_rows[0].get("consensus_var99") or var_rows[0].get("var_99") or 0.0
            var99_frac = float(cv) / max(equity, 1.0)

        all_limits = _state.limit_checker.all_limit_states(
            equity=equity,
            initial_equity=_state.initial_equity,
            daily_pnl=daily_pnl,
            gross_exposure_frac=metrics["gross_exposure_frac"],
            var99_frac=var99_frac,
            max_position_frac=metrics["max_position_frac"],
            per_symbol_notionals=metrics["per_symbol_notionals"],
        )

        return {
            "limits": [
                {
                    "name": lim.name,
                    "type": lim.limit_type.value,
                    "threshold": lim.threshold,
                    "current_value": round(lim.current_value, 6),
                    "is_breached": lim.is_breached,
                    "action": lim.action.value,
                    "severity": lim.severity,
                }
                for lim in all_limits
            ],
            "n_breached": sum(1 for lim in all_limits if lim.is_breached),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------ #
    # GET /risk/attribution                                                #
    # ------------------------------------------------------------------ #

    @app.get("/risk/attribution")
    async def attribution(days: int = Query(default=7, ge=1, le=365)) -> Dict:
        """
        Returns P&L attribution by factor for the last N days.
        """
        if _state.attribution_report is None:
            raise HTTPException(status_code=503, detail="attribution report not initialised")
        summary = _state.attribution_report.summary_dict(days=days)
        summary["days"] = days
        summary["timestamp"] = datetime.now(timezone.utc).isoformat()
        return summary

    # ------------------------------------------------------------------ #
    # GET /risk/correlation                                                #
    # ------------------------------------------------------------------ #

    @app.get("/risk/correlation")
    async def correlation() -> Dict:
        """
        Returns the current correlation matrix, average correlation,
        stress regime flag, PCA explained variance, and concentration metrics.
        """
        if _state.corr_monitor is None:
            raise HTTPException(status_code=503, detail="correlation monitor not initialised")
        return _state.corr_monitor.correlation_json()

else:
    # Stub so the module can be imported even without fastapi
    app = None  # type: ignore


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not _FASTAPI_AVAILABLE:
        raise ImportError("fastapi and uvicorn are required to run the risk API")
    uvicorn.run(
        "execution.risk.risk_api:app",
        host="0.0.0.0",
        port=_PORT,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
