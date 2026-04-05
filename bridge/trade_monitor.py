"""
bridge/trade_monitor.py

TradeMonitor: real-time monitoring of live trades via SQLite polling.

Watches live_trades.db for new trades, computes running statistics,
fires alerts on anomalies, and writes trade_stats.json every 5 minutes.
Pushes stats to the autonomous loop for Bayesian updating.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parents[1]
_LIVE_TRADES_DB = _REPO_ROOT / "execution" / "live_trades.db"
_TRADE_STATS_FILE = _REPO_ROOT / "bridge" / "trade_stats.json"

_POLL_INTERVAL_S = 10.0         # how often to poll the DB
_STATS_WRITE_INTERVAL_S = 300.0 # write trade_stats.json every 5 minutes

# Alert thresholds
_ALERT_LOSING_STREAK = 10
_ALERT_WIN_RATE_MIN = 0.35
_ALERT_SAMPLE_SIZE = 50
_ALERT_MAX_SINGLE_LOSS_PCT = 0.02   # 2% equity


class TradeMonitor:
    """
    Poll live_trades.db for new trades and compute running stats.

    Alerts on:
      - Losing streak > 10
      - Win rate < 35% in last 50 trades
      - Single trade loss > 2% equity

    Writes trade_stats.json every 5 minutes.
    Calls registered callbacks with new stats for Bayesian loop integration.
    """

    def __init__(
        self,
        trades_db: Path | str | None = None,
        stats_file: Path | str | None = None,
        poll_interval: float = _POLL_INTERVAL_S,
    ) -> None:
        self.trades_db = Path(trades_db) if trades_db else _LIVE_TRADES_DB
        self.stats_file = Path(stats_file) if stats_file else _TRADE_STATS_FILE
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        self.poll_interval = poll_interval

        self._recent: deque = deque(maxlen=200)   # last 200 trades in memory
        self._known_ids: set[str] = set()
        self._last_stats_write: float = 0.0
        self._running = False
        self._callbacks: list[Callable[[dict[str, Any]], None]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Poll live_trades.db and fire alerts as new trades arrive."""
        self._running = True
        self._bootstrap()
        logger.info("TradeMonitor: watching %s (poll every %.0fs)", self.trades_db, self.poll_interval)

        try:
            while self._running:
                new_trades = self._poll_new_trades()
                for trade in new_trades:
                    self._process_trade(trade)

                if new_trades:
                    self._check_alerts()

                if self._should_write_stats():
                    stats = self.compute_stats()
                    self._write_stats(stats)
                    self._dispatch(stats)

                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            logger.info("TradeMonitor: stopped.")

    def stop(self) -> None:
        self._running = False

    def register_callback(self, fn: Callable[[dict[str, Any]], None]) -> None:
        """Register a function to be called whenever stats are refreshed."""
        self._callbacks.append(fn)

    def compute_stats(self, last_n: int = 50) -> dict[str, Any]:
        """Compute and return current running statistics."""
        trades = list(self._recent)
        if not trades:
            return {"n_trades": 0, "status": "no_data"}

        recent = trades[-last_n:]
        pnls = [float(t.get("pnl_pct", 0)) for t in recent]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) if pnls else 0.0
        avg_pnl = float(np.mean(pnls)) if pnls else 0.0
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0

        # Current losing streak
        streak = 0
        for p in reversed(pnls):
            if p < 0:
                streak += 1
            else:
                break

        # Profit factor
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe (simplified, annualised assuming hourly bars)
        if len(pnls) > 2:
            sharpe = (avg_pnl / (float(np.std(pnls)) + 1e-9)) * (252 * 24) ** 0.5
        else:
            sharpe = 0.0

        return {
            "n_trades": len(trades),
            "sample_size": len(recent),
            "win_rate": round(win_rate, 4),
            "avg_pnl": round(avg_pnl, 6),
            "avg_win": round(avg_win, 6),
            "avg_loss": round(avg_loss, 6),
            "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else 999.0,
            "losing_streak": streak,
            "sharpe_approx": round(sharpe, 3),
            "gross_profit": round(gross_profit, 6),
            "gross_loss": round(gross_loss, 6),
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "status": "ok",
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _bootstrap(self) -> None:
        """Load last 200 trades on startup to warm up the deque."""
        trades = self._load_trades(limit=200)
        for t in reversed(trades):  # oldest first
            self._recent.append(t)
            self._known_ids.add(str(t.get("trade_id", t.get("rowid", ""))))
        logger.info("TradeMonitor: bootstrapped with %d trades.", len(self._recent))

    def _poll_new_trades(self) -> list[dict[str, Any]]:
        """Return trades that have appeared in the DB since last poll."""
        all_trades = self._load_trades(limit=50)
        new = []
        for t in all_trades:
            tid = str(t.get("trade_id", t.get("rowid", "")))
            if tid not in self._known_ids:
                new.append(t)
                self._known_ids.add(tid)
        return new

    def _process_trade(self, trade: dict[str, Any]) -> None:
        self._recent.append(trade)
        pnl = float(trade.get("pnl_pct", 0))
        sym = trade.get("symbol", "?")
        logger.debug("TradeMonitor: new trade %s pnl=%.4f", sym, pnl)

        # Single trade large loss alert
        if pnl < -_ALERT_MAX_SINGLE_LOSS_PCT:
            logger.warning(
                "TradeMonitor ALERT: large loss %.2f%% on %s", pnl * 100, sym
            )

    def _check_alerts(self) -> None:
        """Check all alert conditions against the current deque."""
        trades = list(self._recent)
        pnls = [float(t.get("pnl_pct", 0)) for t in trades]
        if not pnls:
            return

        # Losing streak
        streak = 0
        for p in reversed(pnls):
            if p < 0:
                streak += 1
            else:
                break
        if streak >= _ALERT_LOSING_STREAK:
            logger.warning(
                "TradeMonitor ALERT: losing streak = %d consecutive losses", streak
            )

        # Win rate in last 50
        recent = pnls[-_ALERT_SAMPLE_SIZE:]
        if len(recent) >= 20:
            wr = sum(1 for p in recent if p > 0) / len(recent)
            if wr < _ALERT_WIN_RATE_MIN:
                logger.warning(
                    "TradeMonitor ALERT: win_rate=%.1f%% < %.1f%% in last %d trades",
                    wr * 100,
                    _ALERT_WIN_RATE_MIN * 100,
                    len(recent),
                )

    def _should_write_stats(self) -> bool:
        import time
        now = time.monotonic()
        if now - self._last_stats_write >= _STATS_WRITE_INTERVAL_S:
            self._last_stats_write = now
            return True
        return False

    def _write_stats(self, stats: dict[str, Any]) -> None:
        try:
            tmp = self.stats_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(stats, indent=2))
            tmp.replace(self.stats_file)
        except Exception as exc:
            logger.warning("TradeMonitor: could not write stats file: %s", exc)

    def _dispatch(self, stats: dict[str, Any]) -> None:
        for cb in self._callbacks:
            try:
                cb(stats)
            except Exception as exc:
                logger.debug("TradeMonitor: callback error: %s", exc)

    def _load_trades(self, limit: int = 200) -> list[dict[str, Any]]:
        if not self.trades_db.exists():
            return []
        try:
            with sqlite3.connect(self.trades_db) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT rowid as trade_id, * FROM trades ORDER BY entry_time DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.debug("TradeMonitor: DB read error: %s", exc)
            return []
