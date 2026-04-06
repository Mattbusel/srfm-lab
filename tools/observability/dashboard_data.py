"""
tools/observability/dashboard_data.py
=======================================
Real-time dashboard data aggregator for the LARSA live trader.

Reads ``execution/live_trades.db`` and the in-process positions state
every 5 seconds to compute a rich dashboard snapshot:

  - Equity curve (last 1000 points) with timestamps
  - Rolling 30-day Sharpe ratio
  - Win rate and profit factor (today + all-time)
  - Per-symbol: unrealised PnL, realised PnL today, position size,
    entry time, hold duration
  - BH state snapshot: which symbols active, which timeframes
  - Open option positions

Serves:
  - ``GET /api/dashboard/snapshot``  — latest JSON snapshot
  - ``WebSocket /ws/dashboard``       — push updates every 5 s

The class is fully thread-safe.  Use ``attach_positions()`` to give it a
live reference to the positions dict from the trader, and
``update_bh_state()`` to push BH activation events.

Usage::

    from tools.observability.dashboard_data import DashboardAggregator

    agg = DashboardAggregator(db_path="execution/live_trades.db", port=8801)
    agg.start()

    # From the trading loop:
    agg.attach_positions(lambda: trader.positions)
    agg.update_bh_state("BTC", "1h", active=True, direction=1, mass=0.72)
    agg.set_equity(102_500.0)

    agg.stop()

Dependencies:
    Standard library + sqlite3 (built-in).
    Optional: numpy (for Sharpe), statistics (built-in fallback).
"""

from __future__ import annotations

import collections
import json
import logging
import math
import socket
import sqlite3
import struct
import threading
import time
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

# ---------------------------------------------------------------------------
# Minimal WebSocket helpers (same pattern as log_streamer.py)
# ---------------------------------------------------------------------------

_WS_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def _ws_handshake(sock: socket.socket, lines: List[bytes]) -> bool:
    import base64, hashlib
    key = None
    for line in lines:
        if line.lower().startswith(b"sec-websocket-key:"):
            key = line.split(b":", 1)[1].strip().decode()
            break
    if not key:
        return False
    accept = base64.b64encode(
        hashlib.sha1((key + _WS_MAGIC).encode()).digest()
    ).decode()
    resp = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )
    try:
        sock.sendall(resp.encode())
        return True
    except OSError:
        return False


def _ws_send(sock: socket.socket, msg: str) -> bool:
    try:
        data = msg.encode("utf-8")
        n = len(data)
        if n < 126:
            hdr = bytes([0x81, n])
        elif n < 65536:
            hdr = struct.pack(">BBH", 0x81, 126, n)
        else:
            hdr = struct.pack(">BBQ", 0x81, 127, n)
        sock.sendall(hdr + data)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _sharpe(returns: List[float], ann_factor: float = 252.0) -> float:
    """Annualised Sharpe from a list of daily returns (fraction)."""
    if len(returns) < 2:
        return 0.0
    if _NUMPY:
        arr = np.array(returns)
        std = float(np.std(arr, ddof=1))
        mean = float(np.mean(arr))
    else:
        n = len(returns)
        mean = sum(returns) / n
        variance = sum((x - mean) ** 2 for x in returns) / (n - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        mean = mean
    if std == 0:
        return 0.0
    return round((mean / std) * math.sqrt(ann_factor), 4)


def _profit_factor(pnls: List[float]) -> float:
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return round(gross_win / gross_loss, 4)


def _win_rate(pnls: List[float]) -> float:
    if not pnls:
        return 0.0
    wins = sum(1 for p in pnls if p > 0)
    return round(wins / len(pnls), 4)


# ---------------------------------------------------------------------------
# DashboardAggregator
# ---------------------------------------------------------------------------

class DashboardAggregator:
    """
    Aggregates live trading state and serves it as a JSON snapshot via HTTP
    and WebSocket.

    All public methods are thread-safe.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        port: int = 8801,
        refresh_interval_secs: float = 5.0,
        equity_curve_max_points: int = 1000,
    ) -> None:
        if db_path is None:
            _repo = Path(__file__).parents[2]
            db_path = str(_repo / "execution" / "live_trades.db")
        self._db_path = Path(db_path)
        self._port = port
        self._refresh = refresh_interval_secs
        self._eq_max = equity_curve_max_points

        self._lock = threading.RLock()
        self._running = False

        # Mutable state
        self._equity: float = 0.0
        self._equity_curve: collections.deque = collections.deque(
            maxlen=equity_curve_max_points
        )
        self._positions_ref: Optional[Callable[[], Dict[str, Any]]] = None

        # BH state: symbol → {timeframe → {active, direction, mass}}
        self._bh_state: Dict[str, Dict[str, Any]] = {}

        # Cached snapshot
        self._last_snapshot: Dict[str, Any] = {}
        self._last_snapshot_time: float = 0.0

        # WebSocket clients
        self._ws_clients: set = set()
        self._ws_lock = threading.Lock()

        # Threads
        self._aggregator_thread: Optional[threading.Thread] = None
        self._http_server: Optional[HTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None
        self._ws_server_thread: Optional[threading.Thread] = None

    # ----------------------------------------------------------------- start/stop
    def start(self) -> None:
        if self._running:
            return
        self._running = True

        self._aggregator_thread = threading.Thread(
            target=self._aggregate_loop,
            name="dashboard-aggregator",
            daemon=True,
        )
        self._aggregator_thread.start()

        self._ws_server_thread = threading.Thread(
            target=self._ws_server_loop,
            name="dashboard-ws",
            daemon=True,
        )
        self._ws_server_thread.start()

        self._start_http_server()
        log.info("DashboardAggregator started on :%d", self._port)

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        with self._ws_lock:
            for sock in list(self._ws_clients):
                try:
                    sock.close()
                except Exception:
                    pass
            self._ws_clients.clear()
        if self._http_server:
            try:
                self._http_server.server_close()
            except Exception:
                pass
        log.info("DashboardAggregator stopped")

    # ----------------------------------------------------------------- setters
    def set_equity(self, equity: float) -> None:
        with self._lock:
            self._equity = equity
            self._equity_curve.append({
                "t": time.time(),
                "equity": round(equity, 2),
            })

    def attach_positions(
        self, positions_fn: Callable[[], Dict[str, Any]]
    ) -> None:
        """Attach a callable that returns the current positions dict."""
        with self._lock:
            self._positions_ref = positions_fn

    def update_bh_state(
        self,
        symbol: str,
        timeframe: str,
        active: bool,
        direction: int = 0,
        mass: float = 0.0,
    ) -> None:
        sym = symbol.upper()
        with self._lock:
            if sym not in self._bh_state:
                self._bh_state[sym] = {}
            self._bh_state[sym][timeframe] = {
                "active": active,
                "direction": direction,
                "mass": round(mass, 4),
                "updated_at": time.time(),
            }

    # ----------------------------------------------------------------- DB queries
    def _query_db(self, sql: str, params: tuple = ()) -> List[Tuple]:
        if not self._db_path.exists():
            return []
        try:
            conn = sqlite3.connect(str(self._db_path), timeout=5.0)
            conn.row_factory = sqlite3.Row
            cur = conn.execute(sql, params)
            rows = cur.fetchall()
            conn.close()
            return [tuple(r) for r in rows]
        except sqlite3.Error as exc:
            log.debug("DashboardAggregator DB error: %s", exc)
            return []

    def _fetch_equity_curve_db(self) -> List[Dict[str, Any]]:
        """Pull the last 1000 closed-trade equity snapshots from the DB."""
        rows = self._query_db(
            """
            SELECT fill_time, trade_pnl
            FROM live_trades
            WHERE fill_time IS NOT NULL AND trade_pnl IS NOT NULL
            ORDER BY fill_time DESC
            LIMIT 1000
            """,
        )
        if not rows:
            return []
        # Build running equity from the most-recent equity value backward
        with self._lock:
            base_equity = self._equity or 0.0
        points = []
        running = base_equity
        for fill_time, pnl in rows:
            points.append({"t": fill_time, "equity": round(running, 2)})
            running -= (pnl or 0.0)
        points.reverse()
        return points[-1000:]

    def _fetch_pnl_records(self, today_only: bool = False) -> List[float]:
        if today_only:
            today_str = date.today().isoformat()
            rows = self._query_db(
                "SELECT trade_pnl FROM live_trades "
                "WHERE trade_pnl IS NOT NULL AND fill_time >= ?",
                (today_str,),
            )
        else:
            rows = self._query_db(
                "SELECT trade_pnl FROM live_trades WHERE trade_pnl IS NOT NULL"
            )
        return [float(r[0]) for r in rows if r[0] is not None]

    def _fetch_daily_returns(self, days: int = 30) -> List[float]:
        """Compute daily PnL fractions for Sharpe calculation."""
        rows = self._query_db(
            """
            SELECT date(fill_time) as d, SUM(trade_pnl) as daily_pnl
            FROM live_trades
            WHERE fill_time IS NOT NULL AND trade_pnl IS NOT NULL
            GROUP BY d
            ORDER BY d DESC
            LIMIT ?
            """,
            (days,),
        )
        if not rows:
            return []
        with self._lock:
            eq = self._equity or 1.0
        return [float(r[1]) / max(eq, 1.0) for r in rows if r[1] is not None]

    def _fetch_per_symbol_pnl(self) -> Dict[str, Dict[str, float]]:
        today_str = date.today().isoformat()
        rows = self._query_db(
            """
            SELECT symbol, SUM(trade_pnl) as total_pnl,
                   SUM(CASE WHEN fill_time >= ? THEN trade_pnl ELSE 0 END) as today_pnl,
                   COUNT(*) as trades
            FROM live_trades
            WHERE trade_pnl IS NOT NULL
            GROUP BY symbol
            """,
            (today_str,),
        )
        result: Dict[str, Dict[str, float]] = {}
        for sym, total_pnl, today_pnl, trades in rows:
            result[str(sym)] = {
                "realized_pnl_total": round(float(total_pnl or 0), 2),
                "realized_pnl_today": round(float(today_pnl or 0), 2),
                "closed_trades": int(trades or 0),
            }
        return result

    def _fetch_open_positions_db(self) -> Dict[str, Dict[str, Any]]:
        """Fetch the most-recent open (unfilled exit) trades from DB."""
        rows = self._query_db(
            """
            SELECT symbol, side, qty, fill_price, fill_time
            FROM live_trades
            WHERE trade_pnl IS NULL
            ORDER BY fill_time ASC
            """,
        )
        positions: Dict[str, Any] = {}
        for sym, side, qty, fill_price, fill_time in rows:
            sym = str(sym)
            positions[sym] = {
                "side": side,
                "qty": float(qty or 0),
                "entry_price": float(fill_price or 0),
                "entry_time": fill_time,
                "hold_minutes": None,
            }
            if fill_time:
                try:
                    entry_ts = float(fill_time)
                    positions[sym]["hold_minutes"] = round(
                        (time.time() - entry_ts) / 60.0, 1
                    )
                except (ValueError, TypeError):
                    pass
        return positions

    # ----------------------------------------------------------------- snapshot builder
    def _build_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            equity = self._equity
            eq_curve_live = list(self._equity_curve)
            bh_state = {
                sym: dict(tfs)
                for sym, tfs in self._bh_state.items()
            }
            positions_fn = self._positions_ref

        # Equity curve: prefer live deque; fall back to DB if empty
        if len(eq_curve_live) < 2:
            eq_curve = self._fetch_equity_curve_db()
        else:
            eq_curve = eq_curve_live

        # PnL stats
        pnl_all = self._fetch_pnl_records(today_only=False)
        pnl_today = self._fetch_pnl_records(today_only=True)
        daily_returns = self._fetch_daily_returns(days=30)

        win_rate_all = _win_rate(pnl_all)
        win_rate_today = _win_rate(pnl_today)
        pf_all = _profit_factor(pnl_all)
        pf_today = _profit_factor(pnl_today)
        sharpe_30d = _sharpe(daily_returns)

        # Per-symbol realised PnL from DB
        sym_pnl = self._fetch_per_symbol_pnl()

        # Open positions: try live reference first, then DB
        if positions_fn is not None:
            try:
                live_pos = positions_fn()
            except Exception:
                live_pos = {}
        else:
            live_pos = {}

        db_open = self._fetch_open_positions_db()
        # Merge: live reference wins
        open_positions: Dict[str, Any] = {}
        for sym, pos_info in db_open.items():
            open_positions[sym] = pos_info
        for sym, pos in live_pos.items():
            # Normalise whatever dict format the trader uses
            if isinstance(pos, dict):
                open_positions[sym] = {
                    "side": pos.get("side", "unknown"),
                    "qty": float(pos.get("qty", pos.get("quantity", 0))),
                    "entry_price": float(pos.get("entry_price", pos.get("avg_entry_price", 0))),
                    "entry_time": pos.get("entry_time", pos.get("entry_ts")),
                    "unrealized_pnl": float(pos.get("unrealized_pl", pos.get("unrealized_pnl", 0))),
                    "current_price": float(pos.get("current_price", pos.get("last_price", 0))),
                    "market_value": float(pos.get("market_value", 0)),
                }
            # Merge per-symbol pnl
            if sym in sym_pnl:
                open_positions[sym].update(sym_pnl[sym])

        # BH active summary
        bh_active_symbols = [
            sym for sym, tfs in bh_state.items()
            if any(tf.get("active") for tf in tfs.values())
        ]

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "equity": round(equity, 2),
            "equity_curve": eq_curve[-self._eq_max:],
            "stats": {
                "sharpe_30d": sharpe_30d,
                "win_rate_alltime": win_rate_all,
                "win_rate_today": win_rate_today,
                "profit_factor_alltime": pf_all,
                "profit_factor_today": pf_today,
                "total_closed_trades": len(pnl_all),
                "closed_trades_today": len(pnl_today),
                "total_realized_pnl": round(sum(pnl_all), 2),
                "realized_pnl_today": round(sum(pnl_today), 2),
            },
            "open_positions": open_positions,
            "bh_state": bh_state,
            "bh_active_symbols": bh_active_symbols,
        }

    # ----------------------------------------------------------------- aggregation loop
    def _aggregate_loop(self) -> None:
        while self._running:
            t0 = time.time()
            try:
                snapshot = self._build_snapshot()
                with self._lock:
                    self._last_snapshot = snapshot
                    self._last_snapshot_time = t0

                payload = json.dumps(snapshot, default=str)
                self._ws_broadcast(payload)
            except Exception as exc:
                log.warning("DashboardAggregator aggregate error: %s", exc)

            elapsed = time.time() - t0
            sleep_for = max(0.0, self._refresh - elapsed)
            # Sleep in small chunks so stop() is responsive
            deadline = time.time() + sleep_for
            while self._running and time.time() < deadline:
                time.sleep(min(0.5, deadline - time.time()))

    # ----------------------------------------------------------------- WebSocket server
    def _ws_broadcast(self, message: str) -> None:
        with self._ws_lock:
            dead = set()
            for sock in self._ws_clients:
                if not _ws_send(sock, message):
                    dead.add(sock)
            for sock in dead:
                self._ws_clients.discard(sock)
                try:
                    sock.close()
                except Exception:
                    pass

    def _ws_server_loop(self) -> None:
        ws_port = self._port + 1  # e.g. :8802
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("0.0.0.0", ws_port))
            srv.listen(10)
            srv.settimeout(1.0)
        except OSError as exc:
            log.error("DashboardAggregator WS bind failed :%d — %s", ws_port, exc)
            return

        log.info("DashboardAggregator WS on :%d/ws/dashboard", ws_port)

        while self._running:
            try:
                client, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._ws_client_handler,
                args=(client,),
                daemon=True,
            ).start()

        try:
            srv.close()
        except Exception:
            pass

    def _ws_client_handler(self, sock: socket.socket) -> None:
        sock.settimeout(60.0)
        try:
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = sock.recv(1024)
                if not chunk:
                    return
                data += chunk
            lines = data.split(b"\r\n")
            req = lines[0].decode("utf-8", errors="replace")
            if "/ws/dashboard" not in req:
                sock.close()
                return
            if not _ws_handshake(sock, lines[1:]):
                sock.close()
                return

            with self._ws_lock:
                self._ws_clients.add(sock)

            # Send current snapshot immediately on connect
            with self._lock:
                snap = dict(self._last_snapshot)
            if snap:
                _ws_send(sock, json.dumps(snap, default=str))

            sock.settimeout(5.0)
            while self._running:
                try:
                    d = sock.recv(128)
                    if not d:
                        break
                except socket.timeout:
                    continue
                except OSError:
                    break
        except Exception:
            pass
        finally:
            with self._ws_lock:
                self._ws_clients.discard(sock)
            try:
                sock.close()
            except Exception:
                pass

    # ----------------------------------------------------------------- HTTP server
    def _start_http_server(self) -> None:
        agg_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?")[0].rstrip("/")

                if path == "/api/dashboard/snapshot":
                    with agg_ref._lock:
                        snap = dict(agg_ref._last_snapshot)
                    if not snap:
                        snap = {"status": "not ready"}
                        status = 503
                    else:
                        status = 200
                    body = json.dumps(snap, indent=2, default=str).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    body = json.dumps({"error": "not found"}).encode()
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

        try:
            httpd = HTTPServer(("0.0.0.0", self._port), Handler)
            httpd.timeout = 1.0
            self._http_server = httpd
        except OSError as exc:
            log.error("DashboardAggregator HTTP bind failed :%d — %s",
                      self._port, exc)
            return

        def _serve():
            log.info("DashboardAggregator HTTP on :%d/api/dashboard/snapshot",
                     self._port)
            while self._running:
                try:
                    httpd.handle_request()
                except Exception:
                    pass

        self._http_thread = threading.Thread(
            target=_serve, name="dashboard-http", daemon=True
        )
        self._http_thread.start()

    # ----------------------------------------------------------------- context manager
    def __enter__(self) -> "DashboardAggregator":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    def get_snapshot(self) -> Dict[str, Any]:
        """Return the most recently computed snapshot dict (thread-safe)."""
        with self._lock:
            return dict(self._last_snapshot)

    def __repr__(self) -> str:
        return (
            f"<DashboardAggregator port={self._port} "
            f"running={self._running} "
            f"ws_clients={len(self._ws_clients)}>"
        )
