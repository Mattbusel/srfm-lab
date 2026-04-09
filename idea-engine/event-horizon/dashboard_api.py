"""
Event Horizon Dashboard API: real-time data feed for the monitoring dashboard.

Serves ALL system state as structured JSON through HTTP endpoints:
  - Real-time equity curve and P&L
  - Per-module health and status
  - Signal heatmap (all signals across all symbols)
  - Regime timeline
  - Consciousness state and belief history
  - Fear/greed gauge with history
  - Groupthink alerts
  - Dream fragility scores
  - Swarm voting results
  - Strategy genome lifecycle
  - Market topology graph data
  - Quantum state visualization
  - Trade log with provenance
  - Alpha allocation table
  - Tear sheet data
  - Guardian status and alerts

This is the single data source for the React dashboard.
All endpoints return JSON. CORS enabled for local development.

Port: 11437 (alongside RAG API on 11435 and Signal API on 11436)
"""

from __future__ import annotations
import json
import time
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Any, Dict, List, Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA STORES
# ═══════════════════════════════════════════════════════════════════════════════

class TimeSeriesStore:
    """Rolling time series data for charting."""

    def __init__(self, max_points: int = 2000):
        self.max_points = max_points
        self._series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self._timestamps: deque = deque(maxlen=max_points)

    def append(self, timestamp: float, data: Dict[str, float]) -> None:
        self._timestamps.append(timestamp)
        for key, value in data.items():
            self._series[key].append(value)

    def get_series(self, key: str, n: int = 500) -> List[float]:
        return list(self._series.get(key, []))[-n:]

    def get_timestamps(self, n: int = 500) -> List[float]:
        return list(self._timestamps)[-n:]

    def get_all(self, n: int = 500) -> Dict:
        return {
            "timestamps": self.get_timestamps(n),
            "series": {k: list(v)[-n:] for k, v in self._series.items()},
        }


class AlertStore:
    """Store for system alerts."""

    def __init__(self, max_alerts: int = 500):
        self._alerts: deque = deque(maxlen=max_alerts)

    def add(self, level: str, source: str, message: str) -> None:
        self._alerts.append({
            "timestamp": time.time(),
            "level": level,
            "source": source,
            "message": message,
        })

    def get_recent(self, n: int = 50) -> List[Dict]:
        return list(self._alerts)[-n:]

    def get_by_level(self, level: str) -> List[Dict]:
        return [a for a in self._alerts if a["level"] == level]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: DASHBOARD STATE
# ═══════════════════════════════════════════════════════════════════════════════

class DashboardState:
    """
    Central state manager for the dashboard.
    All modules push their state here, and the API reads from it.
    """

    def __init__(self):
        # Time series
        self.equity_ts = TimeSeriesStore()
        self.signal_ts = TimeSeriesStore()
        self.regime_ts = TimeSeriesStore()
        self.consciousness_ts = TimeSeriesStore()
        self.fear_greed_ts = TimeSeriesStore()

        # Alerts
        self.alerts = AlertStore()

        # Current state (updated by modules)
        self.current: Dict[str, Any] = {
            "system": {
                "name": "SRFM Event Horizon",
                "version": "1.0.0",
                "status": "running",
                "uptime_hours": 0,
                "bar_count": 0,
            },
            "performance": {
                "equity": 1_000_000,
                "pnl_pct": 0.0,
                "pnl_today_pct": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_dd_pct": 0.0,
                "current_dd_pct": 0.0,
                "n_trades": 0,
                "win_rate": 0.0,
            },
            "positions": [],
            "signals": {},        # symbol -> {signal_name: value}
            "regime": {
                "current": "unknown",
                "confidence": 0.0,
                "transition_prob": 0.0,
                "predicted_next": "unknown",
            },
            "consciousness": {
                "belief": "neutral",
                "activation": 0.0,
                "agreement": 0.0,
                "entropy": 1.0,
                "dominant_domain": "none",
                "phase_transition": None,
            },
            "fear_greed": {
                "index": 0,
                "label": "neutral",
                "multiplier": 1.0,
                "trend": 0.0,
                "components": {},
            },
            "groupthink": {
                "consensus": 0.5,
                "dampening_active": False,
                "multiplier": 1.0,
                "consecutive_bars": 0,
            },
            "dreams": {
                "last_session_time": 0,
                "n_dreams_tested": 0,
                "most_fragile_signal": "",
                "most_robust_signal": "",
                "insights_discovered": 0,
            },
            "swarm": {
                "direction": 0,
                "agreement": 0.0,
                "n_long": 0,
                "n_short": 0,
                "n_abstain": 0,
            },
            "strategy_genome": {
                "alive": 0,
                "by_stage": {},
                "best_organism": "",
                "avg_sharpe": 0.0,
            },
            "topology": {
                "n_clusters": 0,
                "avg_correlation": 0.0,
                "stress": 0.0,
                "leaders": [],
            },
            "quantum": {
                "n_collapsed": 0,
                "n_superposition": 0,
                "portfolio_entropy": 0.0,
                "executable_trades": 0,
            },
            "alpha_allocation": {
                "total_allocated_pct": 0.0,
                "n_sources": 0,
                "by_status": {},
                "top_sources": [],
            },
            "guardian": {
                "status": "active",
                "trading_halted": False,
                "total_alerts": 0,
                "critical_alerts": 0,
            },
            "stability": {
                "certified": True,
                "lyapunov": 0.0,
                "kl_divergence": 0.0,
                "parameter_drift": 0.0,
                "score": 1.0,
            },
            "evolution": {
                "generation": 0,
                "best_fitness": 0.0,
                "mutation_rate": 0.15,
                "adversarial_intensity": 0.5,
                "serendipity_event": None,
            },
            "modules": {},       # module_name -> {healthy, priority, latency_ms}
            "commercial": {
                "api_clients": 0,
                "mrr_usd": 0,
                "hypotheses_sold": 0,
                "total_revenue": 0,
            },
        }

        # Trade log
        self._trade_log: deque = deque(maxlen=1000)

    def update(self, section: str, data: Dict) -> None:
        """Update a section of the dashboard state."""
        if section in self.current:
            if isinstance(self.current[section], dict):
                self.current[section].update(data)
            else:
                self.current[section] = data

    def add_trade(self, trade: Dict) -> None:
        self._trade_log.append({**trade, "recorded_at": time.time()})

    def get_trades(self, n: int = 50) -> List[Dict]:
        return list(self._trade_log)[-n:]

    def push_timeseries(self, timestamp: float, data: Dict) -> None:
        """Push time series data for charting."""
        if "equity" in data:
            self.equity_ts.append(timestamp, {"equity": data["equity"],
                                                "drawdown": data.get("drawdown", 0)})
        if "signals" in data:
            self.signal_ts.append(timestamp, data["signals"])
        if "regime" in data:
            self.regime_ts.append(timestamp, {"regime_code": hash(data["regime"]) % 10})
        if "consciousness" in data:
            self.consciousness_ts.append(timestamp, {"activation": data["consciousness"]})
        if "fear_greed" in data:
            self.fear_greed_ts.append(timestamp, {"fg_index": data["fear_greed"]})


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: SIGNAL HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════

class SignalHeatmapGenerator:
    """Generate heatmap data: symbols x signals matrix."""

    def __init__(self):
        self._data: Dict[str, Dict[str, float]] = defaultdict(dict)

    def update(self, symbol: str, signals: Dict[str, float]) -> None:
        self._data[symbol] = signals

    def get_heatmap(self) -> Dict:
        """Return heatmap data for visualization."""
        if not self._data:
            return {"symbols": [], "signal_names": [], "values": []}

        symbols = sorted(self._data.keys())
        all_signals = set()
        for signals in self._data.values():
            all_signals.update(signals.keys())
        signal_names = sorted(all_signals)

        values = []
        for sym in symbols:
            row = [self._data[sym].get(sig, 0.0) for sig in signal_names]
            values.append(row)

        return {
            "symbols": symbols,
            "signal_names": signal_names,
            "values": values,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: HTTP API SERVER
# ═══════════════════════════════════════════════════════════════════════════════

class DashboardAPI:
    """
    HTTP API server for the Event Horizon dashboard.

    Endpoints:
      GET /api/state                  - Complete current state
      GET /api/performance            - Performance metrics
      GET /api/equity?n=500           - Equity curve time series
      GET /api/signals                - Signal heatmap
      GET /api/regime                 - Regime timeline
      GET /api/consciousness          - Consciousness history
      GET /api/fear-greed             - Fear/greed history
      GET /api/trades?n=50            - Recent trades
      GET /api/alerts?n=50            - Recent alerts
      GET /api/modules                - Module health status
      GET /api/positions              - Current positions
      GET /api/topology               - Market topology data
      GET /api/quantum                - Quantum state data
      GET /api/evolution              - Evolution progress
      GET /api/dreams                 - Dream session results
      GET /api/swarm                  - Swarm voting data
      GET /api/alpha-allocation       - Alpha source allocation
      GET /api/guardian               - Guardian status
      GET /api/stability              - Stability certificate
      GET /api/commercial             - Commercial metrics
      GET /api/tear-sheet             - Full tear sheet data
      GET /health                     - Health check
    """

    def __init__(self, state: DashboardState, heatmap: SignalHeatmapGenerator,
                  port: int = 11437):
        self.state = state
        self.heatmap = heatmap
        self.port = port

    def start(self):
        """Start the dashboard API server."""
        state_ref = self.state
        heatmap_ref = self.heatmap

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                params = parse_qs(parsed.query)

                try:
                    if path == "/health":
                        data = {"status": "ok", "uptime": state_ref.current["system"]["uptime_hours"]}
                    elif path == "/api/state":
                        data = state_ref.current
                    elif path == "/api/performance":
                        data = state_ref.current["performance"]
                    elif path == "/api/equity":
                        n = int(params.get("n", ["500"])[0])
                        data = state_ref.equity_ts.get_all(n)
                    elif path == "/api/signals":
                        data = heatmap_ref.get_heatmap()
                    elif path == "/api/regime":
                        data = {
                            "current": state_ref.current["regime"],
                            "history": state_ref.regime_ts.get_all(200),
                        }
                    elif path == "/api/consciousness":
                        data = {
                            "current": state_ref.current["consciousness"],
                            "history": state_ref.consciousness_ts.get_all(200),
                        }
                    elif path == "/api/fear-greed":
                        data = {
                            "current": state_ref.current["fear_greed"],
                            "history": state_ref.fear_greed_ts.get_all(200),
                        }
                    elif path == "/api/trades":
                        n = int(params.get("n", ["50"])[0])
                        data = {"trades": state_ref.get_trades(n)}
                    elif path == "/api/alerts":
                        n = int(params.get("n", ["50"])[0])
                        data = {"alerts": state_ref.alerts.get_recent(n)}
                    elif path == "/api/modules":
                        data = state_ref.current.get("modules", {})
                    elif path == "/api/positions":
                        data = {"positions": state_ref.current.get("positions", [])}
                    elif path == "/api/topology":
                        data = state_ref.current.get("topology", {})
                    elif path == "/api/quantum":
                        data = state_ref.current.get("quantum", {})
                    elif path == "/api/evolution":
                        data = state_ref.current.get("evolution", {})
                    elif path == "/api/dreams":
                        data = state_ref.current.get("dreams", {})
                    elif path == "/api/swarm":
                        data = state_ref.current.get("swarm", {})
                    elif path == "/api/alpha-allocation":
                        data = state_ref.current.get("alpha_allocation", {})
                    elif path == "/api/guardian":
                        data = state_ref.current.get("guardian", {})
                    elif path == "/api/stability":
                        data = state_ref.current.get("stability", {})
                    elif path == "/api/commercial":
                        data = state_ref.current.get("commercial", {})
                    elif path == "/api/tear-sheet":
                        data = {
                            "performance": state_ref.current["performance"],
                            "modules": state_ref.current.get("modules", {}),
                            "guardian": state_ref.current.get("guardian", {}),
                            "stability": state_ref.current.get("stability", {}),
                        }
                    else:
                        self.send_error(404, f"Unknown endpoint: {path}")
                        return

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(json.dumps(data, default=str).encode("utf-8"))

                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def log_message(self, *args):
                pass  # suppress access logs

        print(f"Dashboard API starting on http://localhost:{self.port}")
        print(f"  22 endpoints serving real-time system state")
        HTTPServer(("0.0.0.0", self.port), Handler).serve_forever()
