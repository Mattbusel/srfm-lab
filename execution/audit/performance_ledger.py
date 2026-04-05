"""
execution/audit/performance_ledger.py
=======================================
Daily P&L ledger with per-trade realized P&L, unrealized P&L snapshots,
and attribution by symbol, strategy, and time-of-day bucket.

Data is stored in SQLite for persistence and can be exported to CSV
for tax reporting or further analysis.

Tables
------
    trade_pnl (
        id              INTEGER PRIMARY KEY,
        trade_date      TEXT,
        order_id        TEXT,
        symbol          TEXT,
        strategy_id     TEXT,
        side            TEXT,
        quantity        REAL,
        entry_price     REAL,
        exit_price      REAL,
        realized_pnl    REAL,
        commission_usd  REAL,
        net_pnl         REAL,
        hour_utc        INTEGER,
        recorded_at     TEXT
    )

    unrealized_snapshots (
        id              INTEGER PRIMARY KEY,
        snapshot_time   TEXT,
        equity          REAL,
        total_unrealized REAL,
        positions_json  TEXT
    )
"""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("execution.performance_ledger")

LEDGER_DB_PATH = Path(__file__).parent.parent / "performance.db"


# ---------------------------------------------------------------------------
# PerformanceLedger
# ---------------------------------------------------------------------------

class PerformanceLedger:
    """
    Records realized P&L per trade and periodic unrealized snapshots.

    Parameters
    ----------
    db_path : Path | str | None
        SQLite database file.  Defaults to ``execution/performance.db``.
    """

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self._db_path = Path(db_path) if db_path else LEDGER_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock    = threading.Lock()
        self._conn    = self._connect()
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection / schema
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS trade_pnl (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date      TEXT    NOT NULL,
                order_id        TEXT    NOT NULL,
                symbol          TEXT    NOT NULL,
                strategy_id     TEXT,
                side            TEXT,
                quantity        REAL,
                entry_price     REAL,
                exit_price      REAL,
                realized_pnl    REAL,
                commission_usd  REAL,
                net_pnl         REAL,
                hour_utc        INTEGER,
                recorded_at     TEXT
            );

            CREATE TABLE IF NOT EXISTS unrealized_snapshots (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_time    TEXT    NOT NULL,
                equity           REAL,
                total_unrealized REAL,
                positions_json   TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_trade_date
                ON trade_pnl (trade_date);
            CREATE INDEX IF NOT EXISTS idx_symbol_pnl
                ON trade_pnl (symbol);
            CREATE INDEX IF NOT EXISTS idx_strategy
                ON trade_pnl (strategy_id);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def record_trade(
        self,
        order,
        entry_price: float,
        realized_pnl: float,
    ) -> int:
        """
        Record the P&L for a completed trade (SELL or closing fill).

        Parameters
        ----------
        order : Order
            The FILLED order.
        entry_price : float
            Average entry price used for cost basis (from PositionTracker).
        realized_pnl : float
            Realized P&L for this fill in USD.

        Returns
        -------
        int
            Database row id.
        """
        net_pnl     = realized_pnl - order.commission_usd
        now         = datetime.now(timezone.utc)
        trade_date  = (order.filled_at or now).date().isoformat()
        hour_utc    = (order.filled_at or now).hour

        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO trade_pnl
                    (trade_date, order_id, symbol, strategy_id, side, quantity,
                     entry_price, exit_price, realized_pnl, commission_usd,
                     net_pnl, hour_utc, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_date,
                    order.order_id,
                    order.symbol,
                    order.strategy_id,
                    order.side.value,
                    order.fill_qty,
                    entry_price,
                    order.fill_price,
                    realized_pnl,
                    order.commission_usd,
                    net_pnl,
                    hour_utc,
                    now.isoformat(),
                ),
            )
            self._conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    def record_unrealized_snapshot(
        self,
        equity: float,
        total_unrealized: float,
        positions: dict,
    ) -> int:
        """
        Persist a point-in-time unrealized P&L snapshot.

        Parameters
        ----------
        equity : float
        total_unrealized : float
        positions : dict
            Output of PositionTracker.export_snapshot()["positions"].
        """
        now = datetime.now(timezone.utc)
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO unrealized_snapshots
                    (snapshot_time, equity, total_unrealized, positions_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    now.isoformat(),
                    equity,
                    total_unrealized,
                    json.dumps(positions, default=str),
                ),
            )
            self._conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Attribution queries
    # ------------------------------------------------------------------

    def pnl_by_symbol(self, for_date: Optional[str] = None) -> dict[str, float]:
        """Return net P&L aggregated per symbol."""
        return self._aggregate("symbol", for_date)

    def pnl_by_strategy(self, for_date: Optional[str] = None) -> dict[str, float]:
        """Return net P&L aggregated per strategy_id."""
        return self._aggregate("strategy_id", for_date)

    def pnl_by_hour(self, for_date: Optional[str] = None) -> dict[int, float]:
        """Return net P&L aggregated per UTC hour of the fill."""
        raw = self._aggregate("hour_utc", for_date)
        return {int(k): v for k, v in raw.items()}

    def _aggregate(self, column: str, for_date: Optional[str]) -> dict:
        if for_date:
            cur = self._conn.execute(
                f"SELECT {column}, SUM(net_pnl) FROM trade_pnl "
                f"WHERE trade_date = ? GROUP BY {column}",
                (for_date,),
            )
        else:
            cur = self._conn.execute(
                f"SELECT {column}, SUM(net_pnl) FROM trade_pnl GROUP BY {column}"
            )
        return {row[0]: row[1] for row in cur.fetchall()}

    def daily_summary(self, for_date: Optional[str] = None) -> dict:
        """
        Return a summary of realized P&L for a given date.

        If for_date is None, returns all-time totals.
        """
        where = "WHERE trade_date = ?" if for_date else ""
        params = (for_date,) if for_date else ()
        cur = self._conn.execute(
            f"""
            SELECT
                COUNT(*)                   AS n_trades,
                SUM(realized_pnl)          AS gross_pnl,
                SUM(commission_usd)        AS total_commission,
                SUM(net_pnl)               AS net_pnl,
                SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) AS winners,
                SUM(CASE WHEN net_pnl < 0 THEN 1 ELSE 0 END) AS losers,
                AVG(net_pnl)               AS avg_trade_pnl,
                MAX(net_pnl)               AS best_trade,
                MIN(net_pnl)               AS worst_trade
            FROM trade_pnl {where}
            """,
            params,
        )
        row = cur.fetchone()
        if row is None or row[0] == 0:
            return {"n_trades": 0}
        n, gross, comm, net, wins, losses, avg, best, worst = row
        return {
            "date":            for_date or "all_time",
            "n_trades":        n,
            "gross_pnl":       gross or 0.0,
            "total_commission": comm or 0.0,
            "net_pnl":         net or 0.0,
            "win_rate":        (wins or 0) / max(n, 1),
            "avg_trade_pnl":   avg or 0.0,
            "best_trade":      best or 0.0,
            "worst_trade":     worst or 0.0,
        }

    # ------------------------------------------------------------------
    # CSV export (tax reporting)
    # ------------------------------------------------------------------

    def export_to_csv(
        self,
        output_path: Path,
        for_date: Optional[str] = None,
    ) -> Path:
        """
        Export trade_pnl rows to a CSV file.

        Parameters
        ----------
        output_path : Path
            Destination CSV path.
        for_date : str | None
            Filter to a specific trade date (ISO format).  None = all dates.

        Returns
        -------
        Path
            The written CSV path.
        """
        where  = "WHERE trade_date = ?" if for_date else ""
        params = (for_date,) if for_date else ()
        cur    = self._conn.execute(
            f"SELECT trade_date, order_id, symbol, strategy_id, side, "
            f"quantity, entry_price, exit_price, realized_pnl, commission_usd, "
            f"net_pnl, hour_utc, recorded_at FROM trade_pnl {where} "
            f"ORDER BY recorded_at ASC",
            params,
        )
        rows   = cur.fetchall()
        headers = [
            "trade_date", "order_id", "symbol", "strategy_id", "side",
            "quantity", "entry_price", "exit_price", "realized_pnl",
            "commission_usd", "net_pnl", "hour_utc", "recorded_at",
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        log.info("PerformanceLedger: exported %d rows to %s", len(rows), output_path)
        return output_path

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
