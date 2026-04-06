"""
slippage_tracker.py — Real-time slippage monitoring.

SlippageTracker hooks into live fill callbacks and measures the gap
between the expected (signal) price and the actual fill price.
Maintains rolling statistics per symbol, persists to SQLite, and
optionally exports Prometheus metrics.

Usage
-----
    tracker = SlippageTracker(db_path="slippage.db")
    tracker.record_fill(
        symbol="ETH",
        side="buy",
        expected_price=2500.0,
        fill_price=2502.5,
        qty=1.0,
        notional=2502.5,
        order_id="abc123",
    )
    stats = tracker.stats("ETH")
    print(stats)
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_DEFAULT_DB = _HERE / "slippage.db"

# ---------------------------------------------------------------------------
# Optional Prometheus
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry  # type: ignore
    _HAS_PROM = True
except ImportError:
    _HAS_PROM = False

_ALERT_THRESHOLD_BPS = 50.0   # alert when slippage > 50 bps
_ALERT_MULTIPLIER = 3.0       # alert when > 3x rolling mean
_ROLLING_WINDOW = 100         # trades per symbol in rolling window


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FillRecord:
    """One recorded fill with full slippage breakdown."""
    symbol: str
    side: str
    expected_price: float
    fill_price: float
    qty: float
    notional: float
    order_id: str
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # computed after init
    realized_slippage_bps: float = 0.0
    market_impact_bps: float = 0.0
    spread_cost_bps: float = 0.0
    adverse_selection_bps: float = 0.0


@dataclass
class SymbolStats:
    symbol: str
    n_fills: int
    mean_slippage_bps: float
    median_slippage_bps: float
    p95_slippage_bps: float
    p99_slippage_bps: float
    std_slippage_bps: float
    mean_impact_bps: float
    mean_spread_bps: float
    total_cost_usd: float
    last_slippage_bps: float
    alert_active: bool


# ---------------------------------------------------------------------------
# Spread table (static half-spread estimates in bps)
# ---------------------------------------------------------------------------
_SPREAD_TABLE: Dict[str, float] = {
    "BTC": 1.0, "ETH": 1.5, "SOL": 3.0, "AVAX": 4.0, "LINK": 5.0,
    "UNI": 5.0, "AAVE": 5.0, "CRV": 6.0, "SUSHI": 7.0, "BAT": 8.0,
    "YFI": 5.0, "DOT": 5.0, "XRP": 2.0, "DOGE": 3.0, "LTC": 3.0,
    "BCH": 4.0, "SHIB": 8.0,
}
_DEFAULT_SPREAD_BPS = 5.0
_DEFAULT_EQUITY_SPREAD_BPS = 2.5
_CRYPTO_SYMS = set(_SPREAD_TABLE.keys())


def _half_spread_bps(symbol: str) -> float:
    sym = symbol.upper()
    if sym in _SPREAD_TABLE:
        return _SPREAD_TABLE[sym]
    return _DEFAULT_EQUITY_SPREAD_BPS


def _sqrt_impact_bps(notional: float, adv_usd: float, sigma: float) -> float:
    """Square-root market impact in bps."""
    if adv_usd <= 0 or notional <= 0:
        return 0.0
    return sigma * math.sqrt(notional / adv_usd) * 10_000


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS slippage_log (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                     TEXT    NOT NULL,
    symbol                 TEXT    NOT NULL,
    side                   TEXT    NOT NULL,
    expected_price         REAL    NOT NULL,
    fill_price             REAL    NOT NULL,
    qty                    REAL    NOT NULL,
    notional               REAL    NOT NULL,
    order_id               TEXT,
    realized_slippage_bps  REAL    NOT NULL,
    market_impact_bps      REAL    NOT NULL,
    spread_cost_bps        REAL    NOT NULL,
    adverse_selection_bps  REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_sl_symbol ON slippage_log(symbol);
CREATE INDEX IF NOT EXISTS ix_sl_ts     ON slippage_log(ts);
"""

_INSERT_SQL = """
INSERT INTO slippage_log
    (ts, symbol, side, expected_price, fill_price, qty, notional, order_id,
     realized_slippage_bps, market_impact_bps, spread_cost_bps, adverse_selection_bps)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
"""


# ---------------------------------------------------------------------------
# SlippageTracker
# ---------------------------------------------------------------------------

class SlippageTracker:
    """
    Real-time slippage tracker.

    Parameters
    ----------
    db_path : Path or str
        SQLite database for persistence.  Created on first use.
    alert_callback : callable, optional
        Called with (symbol, slippage_bps, stats) when an alert fires.
    default_adv_usd : float
        Fallback ADV for market impact model (default $50M).
    default_sigma : float
        Fallback daily return volatility (default 3%).
    prometheus_registry : prometheus_client.CollectorRegistry, optional
        If provided, metrics are registered on this registry instead of
        the default global registry.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        alert_callback: Optional[callable] = None,
        default_adv_usd: float = 50_000_000.0,
        default_sigma: float = 0.03,
        prometheus_registry=None,
    ) -> None:
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB
        self.alert_callback = alert_callback or self._default_alert
        self.default_adv_usd = default_adv_usd
        self.default_sigma = default_sigma

        self._lock = threading.Lock()
        # symbol → deque of last N slippage_bps values
        self._windows: Dict[str, Deque[float]] = {}
        # symbol → deque of FillRecord (last N)
        self._records: Dict[str, Deque[FillRecord]] = {}
        # per-symbol ADV / sigma overrides
        self._adv_map: Dict[str, float] = {}
        self._sigma_map: Dict[str, float] = {}
        # alert state
        self._alert_state: Dict[str, bool] = {}

        self._init_db()
        self._init_prometheus(prometheus_registry)

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(_DDL)
            conn.commit()
        finally:
            conn.close()
        log.info("SlippageTracker DB: %s", self.db_path)

    def _init_prometheus(self, registry) -> None:
        if not _HAS_PROM:
            self._prom = None
            return
        try:
            reg = registry  # may be None → uses default
            kwargs = {"registry": reg} if reg is not None else {}
            self._prom_slip_hist = Histogram(
                "slippage_bps",
                "Realized slippage in basis points",
                ["symbol", "side"],
                buckets=[0, 2, 5, 10, 20, 50, 100, 200, 500],
                **kwargs,
            )
            self._prom_impact_gauge = Gauge(
                "market_impact_bps",
                "Market impact estimate in bps",
                ["symbol"],
                **kwargs,
            )
            self._prom_alert_counter = Counter(
                "slippage_alerts_total",
                "Number of slippage alert triggers",
                ["symbol"],
                **kwargs,
            )
            self._prom = True
        except Exception as exc:
            log.debug("Prometheus init failed: %s", exc)
            self._prom = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_market_params(self, symbol: str, adv_usd: float, sigma: float) -> None:
        """Override ADV and daily-vol for a symbol's impact model."""
        with self._lock:
            self._adv_map[symbol.upper()] = adv_usd
            self._sigma_map[symbol.upper()] = sigma

    def record_fill(
        self,
        symbol: str,
        side: str,
        expected_price: float,
        fill_price: float,
        qty: float,
        notional: float,
        order_id: str = "",
        ts: Optional[str] = None,
    ) -> FillRecord:
        """
        Record a fill and compute its slippage decomposition.

        Parameters
        ----------
        symbol         : e.g. "ETH"
        side           : "buy" or "sell"
        expected_price : price at signal time
        fill_price     : actual execution price
        qty            : shares / contracts / coins
        notional       : fill_price * qty (in USD)
        order_id       : broker order id
        ts             : ISO-8601 timestamp; defaults to now()

        Returns
        -------
        FillRecord with all computed fields populated.
        """
        sym = symbol.upper()
        side_lc = side.lower()

        # realized slippage
        if expected_price > 0:
            if side_lc == "buy":
                slip_bps = (fill_price - expected_price) / expected_price * 10_000
            else:
                slip_bps = (expected_price - fill_price) / expected_price * 10_000
        else:
            slip_bps = 0.0

        # component estimates
        spread_bps = _half_spread_bps(sym)
        adv = self._adv_map.get(sym, self.default_adv_usd)
        sigma = self._sigma_map.get(sym, self.default_sigma)
        impact_bps = _sqrt_impact_bps(notional, adv, sigma)
        adverse_sel_bps = max(0.0, slip_bps - spread_bps - impact_bps)

        rec = FillRecord(
            symbol=sym,
            side=side_lc,
            expected_price=expected_price,
            fill_price=fill_price,
            qty=qty,
            notional=notional,
            order_id=order_id,
            ts=ts or datetime.now(timezone.utc).isoformat(),
            realized_slippage_bps=round(slip_bps, 4),
            market_impact_bps=round(impact_bps, 4),
            spread_cost_bps=round(spread_bps, 4),
            adverse_selection_bps=round(adverse_sel_bps, 4),
        )

        with self._lock:
            self._update_window(sym, slip_bps, rec)
            self._persist(rec)
            self._check_alert(sym, slip_bps)

        if self._prom:
            try:
                self._prom_slip_hist.labels(symbol=sym, side=side_lc).observe(slip_bps)
                self._prom_impact_gauge.labels(symbol=sym).set(impact_bps)
            except Exception:
                pass

        return rec

    def stats(self, symbol: str) -> Optional[SymbolStats]:
        """Return rolling stats for a symbol.  Returns None if no fills yet."""
        sym = symbol.upper()
        with self._lock:
            window = self._windows.get(sym)
            if not window:
                return None
            arr = np.array(window)
            records = list(self._records.get(sym, []))

        costs = sum(
            r.notional * r.realized_slippage_bps / 10_000
            for r in records
        )
        impacts = np.array([r.market_impact_bps for r in records])
        spreads = np.array([r.spread_cost_bps for r in records])

        return SymbolStats(
            symbol=sym,
            n_fills=len(arr),
            mean_slippage_bps=float(np.mean(arr)),
            median_slippage_bps=float(np.median(arr)),
            p95_slippage_bps=float(np.percentile(arr, 95)),
            p99_slippage_bps=float(np.percentile(arr, 99)),
            std_slippage_bps=float(np.std(arr)),
            mean_impact_bps=float(np.mean(impacts)) if len(impacts) else 0.0,
            mean_spread_bps=float(np.mean(spreads)) if len(spreads) else 0.0,
            total_cost_usd=float(costs),
            last_slippage_bps=float(arr[-1]),
            alert_active=self._alert_state.get(sym, False),
        )

    def all_stats(self) -> List[SymbolStats]:
        """Return rolling stats for all tracked symbols."""
        with self._lock:
            syms = list(self._windows.keys())
        return [s for sym in syms if (s := self.stats(sym)) is not None]

    def recent_fills(
        self, symbol: Optional[str] = None, n: int = 20
    ) -> List[FillRecord]:
        """Return up to n recent FillRecords, optionally filtered by symbol."""
        with self._lock:
            if symbol:
                sym = symbol.upper()
                recs = list(self._records.get(sym, []))
            else:
                recs = []
                for dq in self._records.values():
                    recs.extend(dq)
                recs.sort(key=lambda r: r.ts, reverse=True)
        return recs[-n:]

    def query_db(
        self,
        symbol: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 1000,
    ) -> list:
        """Query the slippage_log table. Returns list of dicts."""
        clauses = []
        params: list = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        if since:
            clauses.append("ts >= ?")
            params.append(since)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM slippage_log {where} ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_window(self, sym: str, slip_bps: float, rec: FillRecord) -> None:
        if sym not in self._windows:
            self._windows[sym] = deque(maxlen=_ROLLING_WINDOW)
            self._records[sym] = deque(maxlen=_ROLLING_WINDOW)
        self._windows[sym].append(slip_bps)
        self._records[sym].append(rec)

    def _persist(self, rec: FillRecord) -> None:
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            try:
                conn.execute(_INSERT_SQL, (
                    rec.ts, rec.symbol, rec.side,
                    rec.expected_price, rec.fill_price,
                    rec.qty, rec.notional, rec.order_id,
                    rec.realized_slippage_bps,
                    rec.market_impact_bps,
                    rec.spread_cost_bps,
                    rec.adverse_selection_bps,
                ))
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            log.error("Failed to persist fill record: %s", exc)

    def _check_alert(self, sym: str, slip_bps: float) -> None:
        window = self._windows.get(sym)
        if not window or len(window) < 5:
            return

        arr = np.array(window)
        rolling_mean = float(np.mean(arr[:-1])) if len(arr) > 1 else 0.0

        threshold_abs = _ALERT_THRESHOLD_BPS
        threshold_rel = rolling_mean * _ALERT_MULTIPLIER if rolling_mean > 0 else float("inf")
        threshold = min(threshold_abs, threshold_rel) if rolling_mean > 0 else threshold_abs

        should_alert = slip_bps > threshold
        was_alert = self._alert_state.get(sym, False)

        if should_alert and not was_alert:
            self._alert_state[sym] = True
            stats = self.stats(sym)
            self.alert_callback(sym, slip_bps, stats)
            if self._prom:
                try:
                    self._prom_alert_counter.labels(symbol=sym).inc()
                except Exception:
                    pass
        elif not should_alert and was_alert:
            self._alert_state[sym] = False
            log.info("SLIPPAGE CLEAR: %s slippage back to %.2f bps", sym, slip_bps)

    @staticmethod
    def _default_alert(
        symbol: str, slippage_bps: float, stats: Optional[SymbolStats]
    ) -> None:
        mean_str = f"{stats.mean_slippage_bps:.2f}" if stats else "N/A"
        log.warning(
            "SLIPPAGE ALERT: %s spike=%.2f bps  rolling_mean=%s bps  "
            "(threshold=%.0f bps or 3x normal)",
            symbol, slippage_bps, mean_str, _ALERT_THRESHOLD_BPS,
        )

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "SlippageTracker":
        return self

    def __exit__(self, *_) -> None:
        pass  # SQLite connections are closed per-operation


# ---------------------------------------------------------------------------
# Batch loader from live_trades.db (for seeding the tracker historically)
# ---------------------------------------------------------------------------

def seed_from_live_trades(
    tracker: SlippageTracker,
    live_db_path: Optional[Path] = None,
    since: Optional[str] = None,
) -> int:
    """
    Seed the SlippageTracker from historical live_trades.db records.

    Because live_trades does not store the expected/signal price, this uses
    the first-fill VWAP of each order as the arrival price estimate.

    Returns the number of fills recorded.
    """
    from .tca import _DB_PATH as _TCA_DB

    db = Path(live_db_path) if live_db_path else _TCA_DB
    if not db.exists():
        log.error("live_trades.db not found at %s", db)
        return 0

    conn = sqlite3.connect(db)
    try:
        clauses = []
        params: list = []
        if since:
            clauses.append("fill_time >= ?")
            params.append(since)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        df_rows = conn.execute(
            f"SELECT id, symbol, side, qty, price, notional, fill_time, order_id "
            f"FROM live_trades {where} ORDER BY fill_time",
            params,
        ).fetchall()
    finally:
        conn.close()

    if not df_rows:
        return 0

    import pandas as pd
    df = pd.DataFrame(df_rows, columns=["id", "symbol", "side", "qty", "price",
                                         "notional", "fill_time", "order_id"])
    df["fill_time"] = pd.to_datetime(df["fill_time"], utc=True)
    # compute arrival price per order_id
    arrival: Dict[str, float] = {}
    for oid, grp in df.groupby("order_id"):
        first3 = grp.nsmallest(3, "fill_time")
        vwap_arr = (first3["price"] * first3["qty"]).sum() / first3["qty"].sum()
        arrival[str(oid)] = float(vwap_arr)

    n = 0
    for _, row in df.iterrows():
        oid = str(row["order_id"])
        arr_px = arrival.get(oid, float(row["price"]))
        tracker.record_fill(
            symbol=str(row["symbol"]),
            side=str(row["side"]),
            expected_price=arr_px,
            fill_price=float(row["price"]),
            qty=float(row["qty"]),
            notional=float(row["notional"]),
            order_id=oid,
            ts=str(row["fill_time"]),
        )
        n += 1

    log.info("Seeded SlippageTracker with %d fills from %s", n, db)
    return n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    p = argparse.ArgumentParser(description="SlippageTracker — seed from DB and print stats")
    p.add_argument("--db", default=str(_DEFAULT_DB))
    p.add_argument("--since", default=None, help="ISO-8601 date filter")
    p.add_argument("--symbol", default=None)
    args = p.parse_args()

    tracker = SlippageTracker(db_path=Path(args.db))
    n = seed_from_live_trades(tracker, since=args.since)
    print(f"Seeded {n} fills\n")

    all_stats = tracker.all_stats()
    if args.symbol:
        all_stats = [s for s in all_stats if s.symbol == args.symbol.upper()]

    print(f"{'Symbol':<8} {'N':>6} {'Mean bps':>10} {'P95 bps':>10} {'P99 bps':>10} "
          f"{'Cost $':>12} {'Alert':>6}")
    print("-" * 70)
    for s in sorted(all_stats, key=lambda x: abs(x.total_cost_usd), reverse=True):
        alert_str = "YES" if s.alert_active else "-"
        print(f"{s.symbol:<8} {s.n_fills:>6} {s.mean_slippage_bps:>10.2f} "
              f"{s.p95_slippage_bps:>10.2f} {s.p99_slippage_bps:>10.2f} "
              f"{s.total_cost_usd:>12.2f} {alert_str:>6}")
