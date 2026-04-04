"""
trade_logger.py — SQLite trade log + rolling analytics for LARSA live trader.

Schema mirrors warehouse/schema/03_trades.sql.
The metrics_server reads from TradeLogger to populate Prometheus/InfluxDB gauges.

Usage:
    logger = TradeLogger("data/trades.db")
    logger.log_trade("BTC", "buy", 0.01, 45000, 150.0, 44800, 3)
    logger.log_equity_snapshot(100_000, {"BTC": 0.25, "ETH": 0.10})
    df = logger.get_recent_trades(50)
    stats = logger.get_rolling_stats(20)
"""

from __future__ import annotations

import math
import sqlite3
import threading
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

log = logging.getLogger("larsa.trade_logger")

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------
_DDL_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,                  -- ISO-8601 UTC
    symbol        TEXT    NOT NULL,
    side          TEXT    NOT NULL,                  -- buy | sell
    qty           REAL    NOT NULL,
    price         REAL    NOT NULL,
    entry_price   REAL,
    pnl           REAL    NOT NULL DEFAULT 0.0,
    bars_held     INTEGER NOT NULL DEFAULT 0,
    equity_after  REAL,
    trade_duration_s  REAL,                          -- seconds in position
    notes         TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_ts     ON trades(ts);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
"""

_DDL_EQUITY = """
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    equity    REAL NOT NULL,
    positions TEXT NOT NULL   -- JSON blob: {symbol: fraction}
);
CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_snapshots(ts);
"""

_DDL_REGIME = """
CREATE TABLE IF NOT EXISTS regime_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    d_bh_mass   REAL,
    h_bh_mass   REAL,
    m15_bh_mass REAL,
    d_bh_active INTEGER,
    h_bh_active INTEGER,
    m15_bh_active INTEGER,
    tf_score    INTEGER,
    delta_score REAL,
    atr         REAL,
    garch_vol   REAL,
    ou_zscore   REAL,
    ou_halflife REAL
);
CREATE INDEX IF NOT EXISTS idx_regime_ts     ON regime_log(ts);
CREATE INDEX IF NOT EXISTS idx_regime_symbol ON regime_log(symbol);
"""


# ---------------------------------------------------------------------------
# TradeLogger
# ---------------------------------------------------------------------------
class TradeLogger:
    """Thread-safe SQLite trade + equity log with rolling analytics."""

    def __init__(self, db_path: str = "data/larsa_trades.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = self._open()
        self._init_schema()

    # ------------------------------------------------------------------ #
    # Connection helpers                                                    #
    # ------------------------------------------------------------------ #

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._lock:
            self._conn.executescript(_DDL_TRADES)
            self._conn.executescript(_DDL_EQUITY)
            self._conn.executescript(_DDL_REGIME)
            self._conn.commit()

    def _execute(self, sql: str, params=()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _fetchall(self, sql: str, params=()) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # ------------------------------------------------------------------ #
    # Write methods                                                         #
    # ------------------------------------------------------------------ #

    def log_trade(
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
        notes: Optional[str] = None,
    ):
        """Record a completed trade."""
        ts = datetime.now(timezone.utc).isoformat()
        self._execute(
            """
            INSERT INTO trades
                (ts, symbol, side, qty, price, entry_price, pnl,
                 bars_held, equity_after, trade_duration_s, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                ts, symbol, side, float(qty), float(price),
                float(entry_price) if entry_price is not None else None,
                float(pnl), int(bars_held),
                float(equity_after) if equity_after is not None else None,
                float(trade_duration_s) if trade_duration_s is not None else None,
                notes,
            ),
        )
        log.debug(f"Logged trade: {side} {qty} {symbol} @ {price:.4f}  pnl={pnl:+.2f}")

    def log_equity_snapshot(self, equity: float, positions: Dict[str, float]):
        """Snapshot current equity + position fractions (call every ~15m)."""
        import json
        ts = datetime.now(timezone.utc).isoformat()
        self._execute(
            "INSERT INTO equity_snapshots (ts, equity, positions) VALUES (?,?,?)",
            (ts, float(equity), json.dumps(positions)),
        )

    def log_regime(
        self,
        symbol: str,
        *,
        d_bh_mass: float = 0.0,
        h_bh_mass: float = 0.0,
        m15_bh_mass: float = 0.0,
        d_bh_active: bool = False,
        h_bh_active: bool = False,
        m15_bh_active: bool = False,
        tf_score: int = 0,
        delta_score: float = 0.0,
        atr: float = 0.0,
        garch_vol: float = 0.0,
        ou_zscore: float = 0.0,
        ou_halflife: float = 0.0,
    ):
        ts = datetime.now(timezone.utc).isoformat()
        self._execute(
            """
            INSERT INTO regime_log
                (ts, symbol, d_bh_mass, h_bh_mass, m15_bh_mass,
                 d_bh_active, h_bh_active, m15_bh_active,
                 tf_score, delta_score, atr, garch_vol, ou_zscore, ou_halflife)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                ts, symbol,
                float(d_bh_mass), float(h_bh_mass), float(m15_bh_mass),
                int(d_bh_active), int(h_bh_active), int(m15_bh_active),
                int(tf_score), float(delta_score),
                float(atr), float(garch_vol), float(ou_zscore), float(ou_halflife),
            ),
        )

    # ------------------------------------------------------------------ #
    # Read / analytics                                                      #
    # ------------------------------------------------------------------ #

    def get_recent_trades(self, n: int = 50) -> pd.DataFrame:
        """Return the n most-recent trades as a DataFrame."""
        rows = self._fetchall(
            "SELECT * FROM trades ORDER BY ts DESC LIMIT ?", (n,)
        )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "id", "ts", "symbol", "side", "qty", "price",
                    "entry_price", "pnl", "bars_held", "equity_after",
                    "trade_duration_s", "notes",
                ]
            )
        df = pd.DataFrame([dict(r) for r in rows])
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    def get_rolling_stats(self, window: int = 20) -> Dict[str, float]:
        """
        Compute rolling statistics over the last `window` trades.

        Returns:
            win_rate   — fraction of winning trades  [0, 1]
            sharpe     — annualised Sharpe ratio of per-trade P&L
            avg_pnl    — average P&L per trade
            max_dd     — maximum drawdown over the window (dollar terms)
            total_pnl  — sum of P&L in window
            trade_count — number of trades included
        """
        rows = self._fetchall(
            "SELECT pnl FROM trades ORDER BY ts DESC LIMIT ?", (window,)
        )
        if not rows:
            return {
                "win_rate": 0.0,
                "sharpe": 0.0,
                "avg_pnl": 0.0,
                "max_dd": 0.0,
                "total_pnl": 0.0,
                "trade_count": 0,
            }

        pnls = np.array([r["pnl"] for r in rows], dtype=float)[::-1]  # oldest first
        n = len(pnls)
        wins = float(np.sum(pnls > 0))
        win_rate = wins / n if n > 0 else 0.0

        avg_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls, ddof=1)) if n > 1 else 1e-9
        # Annualise: assume ~8 trades/day
        sharpe = avg_pnl / (std_pnl + 1e-9) * math.sqrt(8 * 252)

        # Max drawdown on cumulative P&L
        cum = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cum)
        dd = running_max - cum
        max_dd = float(np.max(dd))

        return {
            "win_rate": win_rate,
            "sharpe": sharpe,
            "avg_pnl": avg_pnl,
            "max_dd": max_dd,
            "total_pnl": float(np.sum(pnls)),
            "trade_count": n,
        }

    def get_pnl_by_symbol(self) -> pd.DataFrame:
        """Aggregate P&L by symbol for all trades."""
        rows = self._fetchall(
            """
            SELECT symbol,
                   COUNT(*)          AS trade_count,
                   SUM(pnl)          AS total_pnl,
                   AVG(pnl)          AS avg_pnl,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS win_rate,
                   MAX(pnl)          AS best_trade,
                   MIN(pnl)          AS worst_trade
            FROM   trades
            GROUP  BY symbol
            ORDER  BY total_pnl DESC
            """
        )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "symbol", "trade_count", "total_pnl", "avg_pnl",
                    "win_rate", "best_trade", "worst_trade",
                ]
            )
        return pd.DataFrame([dict(r) for r in rows])

    def get_equity_curve(
        self,
        since: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Return equity snapshots, optionally filtered by start time."""
        if since is None:
            since = datetime.now(timezone.utc) - timedelta(days=90)
        rows = self._fetchall(
            "SELECT ts, equity FROM equity_snapshots WHERE ts >= ? ORDER BY ts",
            (since.isoformat(),),
        )
        if not rows:
            return pd.DataFrame(columns=["ts", "equity"])
        df = pd.DataFrame([dict(r) for r in rows])
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    def get_current_drawdown(self) -> float:
        """Current drawdown from peak equity (last 90 days)."""
        df = self.get_equity_curve()
        if df.empty:
            return 0.0
        eq = df["equity"].values
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / (peak + 1e-9)
        return float(dd[-1])

    def get_rolling_win_rates(self, windows=(20, 50, 100)) -> Dict[str, float]:
        """Return win rates for multiple rolling windows."""
        max_w = max(windows)
        rows = self._fetchall(
            "SELECT pnl FROM trades ORDER BY ts DESC LIMIT ?", (max_w,)
        )
        pnls = np.array([r["pnl"] for r in rows], dtype=float)
        result = {}
        for w in windows:
            sl = pnls[:w]
            result[f"win_rate_{w}"] = float(np.mean(sl > 0)) if len(sl) > 0 else 0.0
        return result

    def get_pnl_by_hour(self) -> pd.DataFrame:
        """Average P&L grouped by hour-of-day (UTC)."""
        rows = self._fetchall(
            """
            SELECT CAST(strftime('%H', ts) AS INTEGER) AS hour,
                   AVG(pnl)    AS avg_pnl,
                   COUNT(*)    AS count,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS win_rate
            FROM   trades
            GROUP  BY hour
            ORDER  BY hour
            """
        )
        if not rows:
            return pd.DataFrame(columns=["hour", "avg_pnl", "count", "win_rate"])
        return pd.DataFrame([dict(r) for r in rows])

    def get_pnl_by_weekday(self) -> pd.DataFrame:
        """Average P&L grouped by day-of-week (0=Sunday … 6=Saturday)."""
        rows = self._fetchall(
            """
            SELECT CAST(strftime('%w', ts) AS INTEGER) AS dow,
                   AVG(pnl)    AS avg_pnl,
                   COUNT(*)    AS count,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS win_rate
            FROM   trades
            GROUP  BY dow
            ORDER  BY dow
            """
        )
        if not rows:
            return pd.DataFrame(columns=["dow", "avg_pnl", "count", "win_rate"])
        return pd.DataFrame([dict(r) for r in rows])

    def get_trade_duration_stats(self) -> Dict[str, float]:
        """Statistics on trade duration (seconds)."""
        rows = self._fetchall(
            "SELECT trade_duration_s FROM trades WHERE trade_duration_s IS NOT NULL"
        )
        if not rows:
            return {"mean_s": 0.0, "median_s": 0.0, "p95_s": 0.0}
        durations = np.array([r["trade_duration_s"] for r in rows], dtype=float)
        return {
            "mean_s":   float(np.mean(durations)),
            "median_s": float(np.median(durations)),
            "p95_s":    float(np.percentile(durations, 95)),
        }

    def get_best_worst_trades(self, n: int = 10) -> Dict[str, pd.DataFrame]:
        """Return top-n best and worst trades."""
        best = self._fetchall(
            "SELECT * FROM trades ORDER BY pnl DESC LIMIT ?", (n,)
        )
        worst = self._fetchall(
            "SELECT * FROM trades ORDER BY pnl ASC LIMIT ?", (n,)
        )
        cols = [
            "id", "ts", "symbol", "side", "qty", "price",
            "entry_price", "pnl", "bars_held", "equity_after",
            "trade_duration_s", "notes",
        ]
        return {
            "best":  pd.DataFrame([dict(r) for r in best],  columns=cols) if best  else pd.DataFrame(columns=cols),
            "worst": pd.DataFrame([dict(r) for r in worst], columns=cols) if worst else pd.DataFrame(columns=cols),
        }

    def get_total_trade_count(self) -> int:
        rows = self._fetchall("SELECT COUNT(*) AS n FROM trades")
        return int(rows[0]["n"]) if rows else 0

    # ------------------------------------------------------------------ #
    # Export                                                                #
    # ------------------------------------------------------------------ #

    def export_to_csv(self, path: str = "data/trades_export.csv"):
        """Export all trades to CSV."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = self._fetchall("SELECT * FROM trades ORDER BY ts")
        if not rows:
            log.warning("No trades to export.")
            return
        df = pd.DataFrame([dict(r) for r in rows])
        df.to_csv(p, index=False)
        log.info(f"Exported {len(df)} trades to {p}")

    def export_to_parquet(self, path: str = "data/trades_export.parquet"):
        """Export all trades to Parquet."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = self._fetchall("SELECT * FROM trades ORDER BY ts")
        if not rows:
            log.warning("No trades to export.")
            return
        df = pd.DataFrame([dict(r) for r in rows])
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df.to_parquet(p, index=False)
        log.info(f"Exported {len(df)} trades to {p}")

    def export_equity_to_parquet(self, path: str = "data/equity_export.parquet"):
        """Export equity snapshots to Parquet."""
        import json
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = self._fetchall("SELECT ts, equity, positions FROM equity_snapshots ORDER BY ts")
        if not rows:
            log.warning("No equity snapshots to export.")
            return
        records = []
        for r in rows:
            pos = json.loads(r["positions"])
            rec = {"ts": r["ts"], "equity": r["equity"]}
            rec.update({f"pos_{k}": v for k, v in pos.items()})
            records.append(rec)
        df = pd.DataFrame(records)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df.to_parquet(p, index=False)
        log.info(f"Exported {len(df)} equity snapshots to {p}")

    # ------------------------------------------------------------------ #
    # Utilities                                                             #
    # ------------------------------------------------------------------ #

    def vacuum(self):
        """Reclaim space by vacuuming the SQLite database."""
        with self._lock:
            self._conn.execute("VACUUM")

    def close(self):
        with self._lock:
            self._conn.close()

    def __repr__(self):
        n = self.get_total_trade_count()
        return f"TradeLogger(db={self.db_path}, trades={n})"


# ---------------------------------------------------------------------------
# Convenience factory used by metrics_server
# ---------------------------------------------------------------------------
_default_logger: Optional[TradeLogger] = None


def get_default_logger(db_path: str = "data/larsa_trades.db") -> TradeLogger:
    """Return a module-level singleton TradeLogger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = TradeLogger(db_path)
    return _default_logger


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random, time as _time

    logging.basicConfig(level=logging.DEBUG)
    tl = TradeLogger(":memory:")

    symbols = ["BTC", "ETH", "SOL", "SPY", "QQQ"]
    equity = 100_000.0

    print("Seeding 60 fake trades…")
    for i in range(60):
        sym   = random.choice(symbols)
        side  = random.choice(["buy", "sell"])
        price = random.uniform(100, 50_000)
        qty   = random.uniform(0.001, 0.5)
        pnl   = random.gauss(10, 300)
        equity += pnl
        tl.log_trade(sym, side, qty, price, pnl,
                     entry_price=price * 0.99,
                     bars_held=random.randint(1, 20),
                     equity_after=equity,
                     trade_duration_s=random.uniform(60, 3600))
        if i % 4 == 0:
            pos = {s: random.uniform(0, 0.3) for s in random.sample(symbols, 3)}
            tl.log_equity_snapshot(equity, pos)

    print("Recent trades:")
    print(tl.get_recent_trades(5).to_string())
    print("\nRolling stats (20):", tl.get_rolling_stats(20))
    print("\nP&L by symbol:")
    print(tl.get_pnl_by_symbol().to_string())
    print("\nDrawdown:", tl.get_current_drawdown())
