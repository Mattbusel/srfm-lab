"""
main.py — FastAPI server for Spacetime Arena.

Routes:
  GET  /api/instruments
  POST /api/backtest
  POST /api/mc
  POST /api/sensitivity
  GET  /api/correlation
  GET  /api/trades
  POST /api/archaeology
  POST /api/report
  WS   /ws/live
  WS   /ws/replay
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
_LIB  = _ROOT / "lib"
sys.path.insert(0, str(_LIB))
sys.path.insert(0, str(_ROOT / "spacetime"))

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    handlers=[
        logging.FileHandler(_LOG_DIR / "api.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("spacetime.api")

# ── Live state file ───────────────────────────────────────────────────────────
LIVE_STATE_PATH = Path(__file__).parent.parent / "cache" / "live_state.json"

# ── Thread pool for CPU-bound tasks ───────────────────────────────────────────
_EXECUTOR = ThreadPoolExecutor(max_workers=4)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Spacetime Arena API",
    description="SRFM research, backtesting, and live monitoring platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class BacktestRequest(BaseModel):
    sym:        str
    source:     str = "yfinance"          # yfinance | alpaca | csv
    start:      str = "2020-01-01"
    end:        str = ""
    long_only:  bool = True
    params:     Optional[Dict[str, Any]] = None
    csv_path:   Optional[str] = None      # for source="csv"
    ticker:     Optional[str] = None      # yfinance ticker if different from sym


class MCRequest(BaseModel):
    trades_json:   List[Dict[str, Any]]
    n_sims:        int  = 10_000
    months:        int  = 12
    regime_aware:  bool = True
    serial_corr:   float = 0.3
    starting_equity: float = 1_000_000.0


class SensitivityRequest(BaseModel):
    sym:        str
    source:     str = "yfinance"
    start:      str = "2020-01-01"
    end:        str = ""
    long_only:  bool = True
    params:     Optional[Dict[str, Any]] = None
    csv_path:   Optional[str] = None


class CorrelationRequest(BaseModel):
    syms:   List[str]
    source: str  = "yfinance"
    start:  str  = "2020-01-01"
    end:    str  = ""


class ArchaeologyRequest(BaseModel):
    csv_path: str
    run_name: str


class ReportRequest(BaseModel):
    run_names:           List[str]
    syms:                Optional[List[str]] = None
    include_mc:          bool = True
    include_sensitivity: bool = True
    backtest_results:    Optional[List[Dict[str, Any]]] = None


class ReplayRequest(BaseModel):
    sym:        str
    from_date:  str
    to_date:    str
    speed_mult: float = 1.0
    source:     str   = "yfinance"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_end(end: str) -> str:
    return end if end else datetime.now().strftime("%Y-%m-%d")


async def _run_in_executor(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_EXECUTOR, fn, *args)


def _load_data(source: str, sym: str, start: str, end: str,
               csv_path: Optional[str] = None, ticker: Optional[str] = None) -> pd.DataFrame:
    from engine.data_loader import load_yfinance, load_alpaca, load_csv

    end = _get_end(end)
    if source == "csv":
        if not csv_path:
            raise HTTPException(400, "csv_path required for source=csv")
        return load_csv(csv_path)
    elif source == "alpaca":
        return load_alpaca(sym, start, end)
    else:
        tk = ticker or sym
        return load_yfinance(tk, start, end)


def _backtest_result_to_dict(result) -> Dict[str, Any]:
    """Serialize BacktestResult to JSON-safe dict."""
    trades = []
    for t in result.trades:
        trades.append({
            "entry_time":      str(t.entry_time),
            "exit_time":       str(t.exit_time),
            "sym":             t.sym,
            "entry_price":     t.entry_price,
            "exit_price":      t.exit_price,
            "pnl":             t.pnl,
            "hold_bars":       t.hold_bars,
            "mfe":             t.mfe,
            "mae":             t.mae,
            "tf_score":        t.tf_score,
            "regime":          t.regime,
            "bh_mass_at_entry": t.bh_mass_at_entry,
        })

    curve = [(str(ts), v) for ts, v in result.equity_curve]

    return {
        "sym":             result.sym,
        "trades":          trades,
        "equity_curve":    curve,
        "mass_series_1d":  result.mass_series_1d[:500],   # cap for JSON size
        "mass_series_1h":  result.mass_series_1h[:500],
        "mass_series_15m": result.mass_series_15m[:500],
        "stats":           result.stats,
    }


def _mc_result_to_dict(mc) -> Dict[str, Any]:
    import numpy as np
    return {
        "blowup_rate":       mc.blowup_rate,
        "median_equity":     mc.median_equity,
        "mean_equity":       mc.mean_equity,
        "pct_5":             mc.pct_5,
        "pct_25":            mc.pct_25,
        "pct_75":            mc.pct_75,
        "pct_95":            mc.pct_95,
        "trades_per_month":  mc.trades_per_month,
        "kelly_fraction":    mc.kelly_fraction,
        "regime_stats":      mc.regime_stats,
        "final_equities_sample": mc.final_equities[:200].tolist(),
        "max_drawdowns_sample":  mc.max_drawdowns[:200].tolist(),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/instruments")
async def get_instruments() -> Dict[str, Any]:
    """Return all instrument configs."""
    from engine.bh_engine import INSTRUMENT_CONFIGS
    return {"instruments": INSTRUMENT_CONFIGS}


@app.post("/api/backtest")
async def run_backtest_route(req: BacktestRequest) -> Dict[str, Any]:
    """Run a BH backtest. Returns full result JSON."""
    try:
        def _run():
            from engine.bh_engine import run_backtest
            df = _load_data(req.source, req.sym, req.start, req.end,
                            req.csv_path, req.ticker)
            return run_backtest(req.sym, df, long_only=req.long_only, params=req.params)

        result = await _run_in_executor(_run)
        return _backtest_result_to_dict(result)

    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error("Backtest error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(500, f"Backtest failed: {e}")


@app.post("/api/mc")
async def run_mc_route(req: MCRequest) -> Dict[str, Any]:
    """Run Monte Carlo simulation on provided trades."""
    try:
        def _run():
            from engine.mc import run_mc, MCConfig
            cfg = MCConfig(
                n_sims=req.n_sims,
                months=req.months,
                regime_aware=req.regime_aware,
                serial_corr=req.serial_corr,
            )
            return run_mc(req.trades_json, starting_equity=req.starting_equity, cfg=cfg)

        result = await _run_in_executor(_run)
        return _mc_result_to_dict(result)

    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error("MC error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(500, f"MC failed: {e}")


@app.post("/api/sensitivity")
async def run_sensitivity_route(req: SensitivityRequest) -> Dict[str, Any]:
    """Run parameter sensitivity analysis."""
    try:
        def _run():
            from engine.sensitivity import run_sensitivity, sensitivity_to_dict
            df = _load_data(req.source, req.sym, req.start, req.end,
                            req.csv_path)
            report = run_sensitivity(req.sym, df, long_only=req.long_only, base_params=req.params)
            return sensitivity_to_dict(report)

        return await _run_in_executor(_run)

    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error("Sensitivity error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(500, f"Sensitivity failed: {e}")


@app.get("/api/correlation")
async def run_correlation_route(
    syms:   str = Query(..., description="Comma-separated symbols"),
    source: str = Query("yfinance"),
    start:  str = Query("2020-01-01"),
    end:    str = Query(""),
) -> Dict[str, Any]:
    """Return BH activation correlation matrices for given symbols."""
    sym_list = [s.strip() for s in syms.split(",") if s.strip()]
    if not sym_list:
        raise HTTPException(400, "No symbols provided")

    try:
        def _run():
            from engine.bh_engine import run_backtest
            from engine.correlation import run_correlation_from_bar_states, correlation_to_dict
            from engine.data_loader import load_yfinance, load_alpaca

            bar_states_per_sym = {}
            for sym in sym_list:
                try:
                    if source == "alpaca":
                        df = load_alpaca(sym, start, _get_end(end))
                    else:
                        df = load_yfinance(sym, start, _get_end(end))
                    result = run_backtest(sym, df)
                    bar_states_per_sym[sym] = result.bar_states
                except Exception as e:
                    logger.warning("Correlation: skipping %s: %s", sym, e)

            if not bar_states_per_sym:
                raise ValueError("No valid instruments for correlation")

            corr = run_correlation_from_bar_states(bar_states_per_sym)
            return correlation_to_dict(corr)

        return await _run_in_executor(_run)

    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error("Correlation error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(500, f"Correlation failed: {e}")


@app.get("/api/trades")
async def get_trades_route(
    sym:          Optional[str] = Query(None),
    from_date:    Optional[str] = Query(None),
    to_date:      Optional[str] = Query(None),
    regime:       Optional[str] = Query(None),
    min_tf_score: Optional[int] = Query(None),
    run_name:     Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Query trades from the archaeology DB."""
    try:
        from engine.archaeology import get_trades
        trades = get_trades(sym=sym, from_date=from_date, to_date=to_date,
                             regime=regime, min_tf_score=min_tf_score, run_name=run_name)
        return {"trades": trades, "count": len(trades)}
    except Exception as e:
        logger.error("Trades query error: %s", e)
        raise HTTPException(500, f"DB query failed: {e}")


@app.post("/api/archaeology")
async def run_archaeology_route(req: ArchaeologyRequest) -> Dict[str, Any]:
    """Parse QC CSV and populate the archaeology DB."""
    try:
        def _run():
            from engine.archaeology import run_archaeology
            inserted = run_archaeology(req.csv_path, req.run_name)
            return len(inserted)

        count = await _run_in_executor(_run)
        return {"status": "ok", "inserted": count, "run_name": req.run_name}

    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.error("Archaeology error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(500, f"Archaeology failed: {e}")


@app.post("/api/report")
async def generate_report_route(req: ReportRequest) -> Dict[str, Any]:
    """Generate PDF report. Returns file path."""
    try:
        def _run():
            from reports.generator import generate_report

            bt_results = req.backtest_results or []
            path = generate_report(
                run_names=req.run_names,
                backtest_results=bt_results,
                include_mc=req.include_mc,
                include_sensitivity=req.include_sensitivity,
            )
            return str(path)

        path = await _run_in_executor(_run)
        return {"status": "ok", "file_path": path}

    except Exception as e:
        logger.error("Report error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(500, f"Report generation failed: {e}")


# ---------------------------------------------------------------------------
# WebSocket: /ws/live
# ---------------------------------------------------------------------------

@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    """
    Stream live trader state every 10s.
    Reads from spacetime/cache/live_state.json which the live trader writes.
    """
    await ws.accept()
    logger.info("WS /ws/live connected")
    try:
        while True:
            payload: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "instruments": {},
                "equity": None,
            }

            if LIVE_STATE_PATH.exists():
                try:
                    with open(LIVE_STATE_PATH) as f:
                        state = json.load(f)
                    payload.update(state)
                except Exception as e:
                    payload["error"] = str(e)
            else:
                payload["note"] = "live_state.json not found — live trader not running"

            await ws.send_json(payload)
            await asyncio.sleep(10)

    except WebSocketDisconnect:
        logger.info("WS /ws/live disconnected")
    except Exception as e:
        logger.error("WS /ws/live error: %s", e)
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WebSocket: /ws/replay
# ---------------------------------------------------------------------------

@app.websocket("/ws/replay")
async def ws_replay(ws: WebSocket) -> None:
    """
    Stream bar-by-bar BH state replay.
    Client sends JSON: {sym, from_date, to_date, speed_mult, source}
    Server streams event dicts.
    """
    await ws.accept()
    logger.info("WS /ws/replay connected")

    try:
        raw = await ws.receive_text()
        req = json.loads(raw)

        sym        = req.get("sym", "ES")
        from_date  = req.get("from_date", "2020-01-01")
        to_date    = req.get("to_date", datetime.now().strftime("%Y-%m-%d"))
        speed_mult = float(req.get("speed_mult", 1.0))
        source     = req.get("source", "yfinance")
        ticker     = req.get("ticker", sym)

        logger.info("WS replay: sym=%s %s to %s speed=%.1fx", sym, from_date, to_date, speed_mult)

        # Load data in thread (I/O bound)
        def _load():
            from engine.data_loader import load_yfinance, load_alpaca
            if source == "alpaca":
                return load_alpaca(sym, from_date, to_date)
            else:
                return load_yfinance(ticker, from_date, to_date)

        df = await _run_in_executor(_load)

        # Stream replay events
        from engine.replay import async_replay_bars
        async for event in async_replay_bars(sym, df, speed_mult=speed_mult):
            await ws.send_json(event)

        await ws.send_json({"status": "replay_complete"})

    except WebSocketDisconnect:
        logger.info("WS /ws/replay disconnected")
    except json.JSONDecodeError as e:
        await ws.send_json({"error": f"Invalid JSON: {e}"})
    except Exception as e:
        logger.error("WS /ws/replay error: %s\n%s", e, traceback.format_exc())
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8765,
        reload=False,
        log_level="info",
    )
