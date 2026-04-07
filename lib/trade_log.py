"""
lib/trade_log.py
================
Structured trade logging with full context capture for the LARSA live trader.

Every trade entry captures 30+ fields including:
  - Price, qty, side
  - All signal values at entry (BH mass, CF scores, GARCH vol, Hurst, ML, RL)
  - Regime state and NAV state at entry
  - Complete parameter snapshot (for post-trade attribution)

Every trade exit captures:
  - Exit price, reason, P&L (absolute + %)
  - Bars held, exit signals, slippage estimate

Writes to SQLite (trade_log and trade_exit tables) plus an in-memory
index for open positions.

Usage:
    from lib.trade_log import get_trade_log

    tl = get_trade_log()
    tid = tl.log_entry("BTC", price=65000, qty=0.01, side="long", context={...})
    tl.log_exit(tid, price=66000, reason="BH_COLLAPSE", pnl=10.0)
    open_pos = tl.get_open_positions()
    trades   = tl.get_closed_trades()
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("trade_log")

_REPO_ROOT  = Path(__file__).parents[1]
_DEFAULT_DB = _REPO_ROOT / "execution" / "live_trades.db"


# ---------------------------------------------------------------------------
# DB schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trade_entries (
    trade_id            TEXT    PRIMARY KEY,
    ts_entry            TEXT    NOT NULL,
    symbol              TEXT    NOT NULL,
    side                TEXT    NOT NULL,
    entry_price         REAL    NOT NULL,
    qty                 REAL    NOT NULL,
    notional            REAL,
    -- Signal values at entry
    bh_mass_15m         REAL,
    bh_mass_1h          REAL,
    bh_mass_4h          REAL,
    bh_dir_15m          INTEGER,
    bh_dir_1h           INTEGER,
    bh_dir_4h           INTEGER,
    cf_bull             REAL,
    cf_bear             REAL,
    cf_score            REAL,
    tf_score            INTEGER,
    garch_vol           REAL,
    garch_vol_scale     REAL,
    hurst_exp           REAL,
    ou_zscore           REAL,
    ml_signal           REAL,
    granger_boost       REAL,
    rl_confidence       REAL,
    nav_omega           REAL,
    nav_omega_ratio     REAL,
    nav_geodesic        REAL,
    nav_geodesic_ratio  REAL,
    -- Regime state
    regime              TEXT,
    corr_mode           TEXT,
    dd_from_peak        REAL,
    event_blocked       INTEGER,
    -- NAV state
    portfolio_nav       REAL,
    crypto_exposure     REAL,
    equity_exposure     REAL,
    -- Parameter snapshot (JSON)
    param_snapshot      TEXT,
    -- Raw context JSON for everything else
    context_json        TEXT
);
CREATE INDEX IF NOT EXISTS idx_te_symbol ON trade_entries(symbol);
CREATE INDEX IF NOT EXISTS idx_te_ts     ON trade_entries(ts_entry);

CREATE TABLE IF NOT EXISTS trade_exits (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id        TEXT    NOT NULL REFERENCES trade_entries(trade_id),
    ts_exit         TEXT    NOT NULL,
    exit_price      REAL    NOT NULL,
    qty_closed      REAL    NOT NULL,
    reason          TEXT    NOT NULL,
    pnl_abs         REAL,
    pnl_pct         REAL,
    bars_held       INTEGER,
    -- Exit signal state
    bh_mass_at_exit REAL,
    tf_score_at_exit INTEGER,
    rl_exit_signal  REAL,
    slippage_est    REAL,
    notes           TEXT
);
CREATE INDEX IF NOT EXISTS idx_tx_trade_id ON trade_exits(trade_id);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_trade_id() -> str:
    """Generate a short unique trade ID: timestamp prefix + 6-char hex."""
    ts  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    uid = uuid.uuid4().hex[:6].upper()
    return f"{ts}_{uid}"


# ---------------------------------------------------------------------------
# TradeEntry dataclass
# ---------------------------------------------------------------------------

@dataclass
class TradeEntry:
    """
    Complete record of a trade entry event.
    All signal values captured at the moment the order was submitted.
    """
    trade_id:           str
    ts_entry:           datetime
    symbol:             str
    side:               str          # "long" | "short"
    entry_price:        float
    qty:                float
    notional:           float        = 0.0

    # Signal values at entry
    bh_mass_15m:        Optional[float] = None
    bh_mass_1h:         Optional[float] = None
    bh_mass_4h:         Optional[float] = None
    bh_dir_15m:         Optional[int]   = None   # +1 / -1 / 0
    bh_dir_1h:          Optional[int]   = None
    bh_dir_4h:          Optional[int]   = None
    cf_bull:            Optional[float] = None
    cf_bear:            Optional[float] = None
    cf_score:           Optional[float] = None
    tf_score:           Optional[int]   = None
    garch_vol:          Optional[float] = None
    garch_vol_scale:    Optional[float] = None
    hurst_exp:          Optional[float] = None
    ou_zscore:          Optional[float] = None
    ml_signal:          Optional[float] = None
    granger_boost:      Optional[float] = None
    rl_confidence:      Optional[float] = None
    nav_omega:          Optional[float] = None
    nav_omega_ratio:    Optional[float] = None
    nav_geodesic:       Optional[float] = None
    nav_geodesic_ratio: Optional[float] = None

    # Regime / portfolio state
    regime:             Optional[str]   = None   # "trending" | "ranging" | "volatile"
    corr_mode:          Optional[str]   = None   # "normal" | "stress"
    dd_from_peak:       Optional[float] = None
    event_blocked:      bool            = False
    portfolio_nav:      Optional[float] = None
    crypto_exposure:    Optional[float] = None
    equity_exposure:    Optional[float] = None

    # Parameter snapshot and raw context
    param_snapshot: dict[str, Any]  = field(default_factory=dict)
    context:        dict[str, Any]  = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ts_entry"] = self.ts_entry.isoformat()
        return d

    def is_long(self) -> bool:
        return self.side.lower() == "long"

    def is_short(self) -> bool:
        return self.side.lower() == "short"


# ---------------------------------------------------------------------------
# TradeExit dataclass
# ---------------------------------------------------------------------------

@dataclass
class TradeExit:
    """Record of a trade exit event."""
    trade_id:         str
    ts_exit:          datetime
    exit_price:       float
    qty_closed:       float
    reason:           str         # "BH_COLLAPSE" | "STOP_LOSS" | "RL_EXIT" | "TIMEOUT" | "MANUAL" | etc.
    pnl_abs:          Optional[float] = None
    pnl_pct:          Optional[float] = None
    bars_held:        Optional[int]   = None
    bh_mass_at_exit:  Optional[float] = None
    tf_score_at_exit: Optional[int]   = None
    rl_exit_signal:   Optional[float] = None
    slippage_est:     Optional[float] = None
    notes:            str             = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ts_exit"] = self.ts_exit.isoformat()
        return d


# ---------------------------------------------------------------------------
# TradeLog
# ---------------------------------------------------------------------------

class TradeLog:
    """
    Thread-safe structured trade log.

    Maintains an in-memory index of open positions (trade_id -> TradeEntry).
    Closed trades are retained in memory for the current session and always
    available from SQLite for historical queries.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path     = Path(db_path) if db_path else _DEFAULT_DB
        self._lock        = threading.RLock()
        self._open:       dict[str, TradeEntry] = {}     # trade_id -> entry
        # closed in this session: trade_id -> (entry, exit)
        self._closed:     dict[str, tuple[TradeEntry, TradeExit]] = {}
        self._conn: Optional[sqlite3.Connection] = None

        self._init_db()

    # ------------------------------------------------------------------
    # Entry logging
    # ------------------------------------------------------------------

    def log_entry(
        self,
        symbol:     str,
        price:      float,
        qty:        float,
        side:       str,
        context:    dict[str, Any],
        trade_id:   Optional[str] = None,
    ) -> str:
        """
        Record a trade entry.

        context dict keys (all optional):
            bh_mass_15m, bh_mass_1h, bh_mass_4h,
            bh_dir_15m, bh_dir_1h, bh_dir_4h,
            cf_bull, cf_bear, cf_score, tf_score,
            garch_vol, garch_vol_scale, hurst_exp, ou_zscore,
            ml_signal, granger_boost, rl_confidence,
            nav_omega, nav_omega_ratio, nav_geodesic, nav_geodesic_ratio,
            regime, corr_mode, dd_from_peak, event_blocked,
            portfolio_nav, crypto_exposure, equity_exposure,
            param_snapshot (dict)

        Returns the trade_id string.
        """
        tid = trade_id or _new_trade_id()
        g   = context.get

        entry = TradeEntry(
            trade_id            = tid,
            ts_entry            = datetime.now(timezone.utc),
            symbol              = symbol.upper(),
            side                = side.lower(),
            entry_price         = float(price),
            qty                 = float(qty),
            notional            = float(g("notional", price * qty)),
            bh_mass_15m         = _opt_float(g("bh_mass_15m")),
            bh_mass_1h          = _opt_float(g("bh_mass_1h")),
            bh_mass_4h          = _opt_float(g("bh_mass_4h")),
            bh_dir_15m          = _opt_int(g("bh_dir_15m")),
            bh_dir_1h           = _opt_int(g("bh_dir_1h")),
            bh_dir_4h           = _opt_int(g("bh_dir_4h")),
            cf_bull             = _opt_float(g("cf_bull")),
            cf_bear             = _opt_float(g("cf_bear")),
            cf_score            = _opt_float(g("cf_score")),
            tf_score            = _opt_int(g("tf_score")),
            garch_vol           = _opt_float(g("garch_vol")),
            garch_vol_scale     = _opt_float(g("garch_vol_scale")),
            hurst_exp           = _opt_float(g("hurst_exp")),
            ou_zscore           = _opt_float(g("ou_zscore")),
            ml_signal           = _opt_float(g("ml_signal")),
            granger_boost       = _opt_float(g("granger_boost")),
            rl_confidence       = _opt_float(g("rl_confidence")),
            nav_omega           = _opt_float(g("nav_omega")),
            nav_omega_ratio     = _opt_float(g("nav_omega_ratio")),
            nav_geodesic        = _opt_float(g("nav_geodesic")),
            nav_geodesic_ratio  = _opt_float(g("nav_geodesic_ratio")),
            regime              = g("regime"),
            corr_mode           = g("corr_mode"),
            dd_from_peak        = _opt_float(g("dd_from_peak")),
            event_blocked       = bool(g("event_blocked", False)),
            portfolio_nav       = _opt_float(g("portfolio_nav")),
            crypto_exposure     = _opt_float(g("crypto_exposure")),
            equity_exposure     = _opt_float(g("equity_exposure")),
            param_snapshot      = g("param_snapshot") or {},
            context             = {k: v for k, v in context.items() if k != "param_snapshot"},
        )

        with self._lock:
            self._open[tid] = entry

        self._persist_entry(entry)
        log.info(
            "trade_log: ENTRY %s  %s  %s  qty=%.6f  price=%.4f  tf=%s",
            tid, symbol.upper(), side.upper(), qty, price, g("tf_score"),
        )
        return tid

    # ------------------------------------------------------------------
    # Exit logging
    # ------------------------------------------------------------------

    def log_exit(
        self,
        trade_id:    str,
        price:       float,
        reason:      str,
        pnl:         Optional[float]  = None,
        context:     Optional[dict]   = None,
        bars_held:   Optional[int]    = None,
    ) -> Optional[TradeExit]:
        """
        Record a trade exit.

        Computes P&L from entry/exit prices if pnl not provided.
        Moves trade from open to closed in-memory index.
        Returns the TradeExit dataclass.
        """
        with self._lock:
            entry = self._open.get(trade_id)

        if entry is None:
            log.warning("trade_log: log_exit -- trade_id '%s' not found in open positions", trade_id)
            return None

        ctx = context or {}
        g   = ctx.get

        # Compute P&L if not provided
        pnl_abs, pnl_pct = self._compute_pnl(entry, price, pnl)

        exit_rec = TradeExit(
            trade_id         = trade_id,
            ts_exit          = datetime.now(timezone.utc),
            exit_price       = float(price),
            qty_closed       = entry.qty,
            reason           = reason,
            pnl_abs          = pnl_abs,
            pnl_pct          = pnl_pct,
            bars_held        = bars_held or g("bars_held"),
            bh_mass_at_exit  = _opt_float(g("bh_mass_at_exit")),
            tf_score_at_exit = _opt_int(g("tf_score_at_exit")),
            rl_exit_signal   = _opt_float(g("rl_exit_signal")),
            slippage_est     = _opt_float(g("slippage_est")),
            notes            = str(g("notes", "")),
        )

        with self._lock:
            self._open.pop(trade_id, None)
            self._closed[trade_id] = (entry, exit_rec)

        self._persist_exit(exit_rec)

        sign = "+" if (pnl_abs or 0) >= 0 else ""
        log.info(
            "trade_log: EXIT  %s  %s  reason=%-14s  pnl=%s%.2f (%.2f%%)  bars=%s",
            trade_id,
            entry.symbol,
            reason,
            sign,
            pnl_abs or 0,
            pnl_pct or 0,
            bars_held or "?",
        )
        return exit_rec

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[TradeEntry]:
        """Return list of all currently open TradeEntry records."""
        with self._lock:
            return list(self._open.values())

    def get_open_position(self, trade_id: str) -> Optional[TradeEntry]:
        """Return a single open TradeEntry, or None."""
        with self._lock:
            return self._open.get(trade_id)

    def get_open_by_symbol(self, symbol: str) -> list[TradeEntry]:
        """Return open positions for a specific symbol."""
        sym = symbol.upper()
        with self._lock:
            return [e for e in self._open.values() if e.symbol == sym]

    def get_closed_trades(
        self,
        since: Optional[datetime] = None,
        symbol: Optional[str]    = None,
        limit:  int               = 500,
    ) -> list[tuple[TradeEntry, TradeExit]]:
        """
        Return closed trades from the database.

        If since is None returns from session memory first; falls back
        to DB for historical queries. Results are (TradeEntry, TradeExit) pairs.
        """
        # Use in-memory session data when no filters
        if since is None and symbol is None:
            with self._lock:
                pairs = list(self._closed.values())
            return pairs[-limit:]

        # Query DB
        return self._query_closed_trades(since=since, symbol=symbol, limit=limit)

    def get_entry_context(self, trade_id: str) -> dict[str, Any]:
        """
        Retrieve the full entry context for a trade.

        First checks in-memory open/closed; falls back to DB.
        """
        with self._lock:
            entry = self._open.get(trade_id)
            if entry is None and trade_id in self._closed:
                entry = self._closed[trade_id][0]

        if entry is not None:
            return {
                **entry.to_dict(),
                "param_snapshot": entry.param_snapshot,
                "context":        entry.context,
            }

        # DB fallback
        return self._fetch_entry_context_from_db(trade_id)

    def get_pnl_summary(
        self,
        since: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Return P&L summary statistics for closed trades."""
        trades = self.get_closed_trades(since=since, limit=10000)
        if not trades:
            return {"n_trades": 0}

        pnls = [
            tx.pnl_abs for _, tx in trades if tx.pnl_abs is not None
        ]
        pnl_pcts = [
            tx.pnl_pct for _, tx in trades if tx.pnl_pct is not None
        ]
        if not pnls:
            return {"n_trades": len(trades)}

        wins  = [p for p in pnls if p > 0]
        total = sum(pnls)
        avg   = total / len(pnls)

        return {
            "n_trades":      len(trades),
            "n_wins":        len(wins),
            "n_losses":      len(pnls) - len(wins),
            "win_rate":      len(wins) / len(pnls) if pnls else 0.0,
            "total_pnl":     round(total, 2),
            "avg_pnl":       round(avg, 2),
            "avg_win":       round(sum(wins) / len(wins), 2) if wins else 0.0,
            "avg_loss":      round(sum(p for p in pnls if p <= 0) / max(1, len(pnls) - len(wins)), 2),
            "max_win":       round(max(pnls), 2),
            "max_loss":      round(min(pnls), 2),
            "avg_pnl_pct":   round(sum(pnl_pcts) / len(pnl_pcts), 4) if pnl_pcts else None,
        }

    def open_trade_ids(self) -> list[str]:
        """Return sorted list of open trade IDs."""
        with self._lock:
            return sorted(self._open.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_pnl(
        entry: TradeEntry,
        exit_price: float,
        provided_pnl: Optional[float],
    ) -> tuple[Optional[float], Optional[float]]:
        """Compute (pnl_abs, pnl_pct) from entry/exit, or use provided_pnl."""
        if provided_pnl is not None:
            pnl_abs = provided_pnl
        else:
            sign    = 1 if entry.is_long() else -1
            pnl_abs = sign * (exit_price - entry.entry_price) * entry.qty

        pnl_pct = None
        if entry.notional and entry.notional > 0:
            pnl_pct = (pnl_abs / entry.notional) * 100.0
        elif entry.entry_price > 0 and entry.qty > 0:
            cost    = entry.entry_price * entry.qty
            pnl_pct = (pnl_abs / cost) * 100.0

        return pnl_abs, pnl_pct

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
            self._conn = conn
        except Exception as exc:
            log.warning("trade_log: DB init failed -- %s", exc)
            self._conn = None

    def _persist_entry(self, entry: TradeEntry) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute(
                """INSERT OR REPLACE INTO trade_entries (
                    trade_id, ts_entry, symbol, side, entry_price, qty, notional,
                    bh_mass_15m, bh_mass_1h, bh_mass_4h,
                    bh_dir_15m, bh_dir_1h, bh_dir_4h,
                    cf_bull, cf_bear, cf_score, tf_score,
                    garch_vol, garch_vol_scale, hurst_exp, ou_zscore,
                    ml_signal, granger_boost, rl_confidence,
                    nav_omega, nav_omega_ratio, nav_geodesic, nav_geodesic_ratio,
                    regime, corr_mode, dd_from_peak, event_blocked,
                    portfolio_nav, crypto_exposure, equity_exposure,
                    param_snapshot, context_json
                ) VALUES (
                    ?,?,?,?,?,?,?,
                    ?,?,?,?,?,?,
                    ?,?,?,?,
                    ?,?,?,?,
                    ?,?,?,
                    ?,?,?,?,
                    ?,?,?,?,
                    ?,?,?,
                    ?,?
                )""",
                (
                    entry.trade_id,
                    entry.ts_entry.isoformat(),
                    entry.symbol,
                    entry.side,
                    entry.entry_price,
                    entry.qty,
                    entry.notional,
                    entry.bh_mass_15m,
                    entry.bh_mass_1h,
                    entry.bh_mass_4h,
                    entry.bh_dir_15m,
                    entry.bh_dir_1h,
                    entry.bh_dir_4h,
                    entry.cf_bull,
                    entry.cf_bear,
                    entry.cf_score,
                    entry.tf_score,
                    entry.garch_vol,
                    entry.garch_vol_scale,
                    entry.hurst_exp,
                    entry.ou_zscore,
                    entry.ml_signal,
                    entry.granger_boost,
                    entry.rl_confidence,
                    entry.nav_omega,
                    entry.nav_omega_ratio,
                    entry.nav_geodesic,
                    entry.nav_geodesic_ratio,
                    entry.regime,
                    entry.corr_mode,
                    entry.dd_from_peak,
                    int(entry.event_blocked),
                    entry.portfolio_nav,
                    entry.crypto_exposure,
                    entry.equity_exposure,
                    json.dumps(entry.param_snapshot) if entry.param_snapshot else None,
                    json.dumps(entry.context) if entry.context else None,
                ),
            )
            self._conn.commit()
        except Exception as exc:
            log.error("trade_log: persist_entry error -- %s", exc)

    def _persist_exit(self, exit_rec: TradeExit) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute(
                """INSERT INTO trade_exits (
                    trade_id, ts_exit, exit_price, qty_closed, reason,
                    pnl_abs, pnl_pct, bars_held,
                    bh_mass_at_exit, tf_score_at_exit, rl_exit_signal,
                    slippage_est, notes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    exit_rec.trade_id,
                    exit_rec.ts_exit.isoformat(),
                    exit_rec.exit_price,
                    exit_rec.qty_closed,
                    exit_rec.reason,
                    exit_rec.pnl_abs,
                    exit_rec.pnl_pct,
                    exit_rec.bars_held,
                    exit_rec.bh_mass_at_exit,
                    exit_rec.tf_score_at_exit,
                    exit_rec.rl_exit_signal,
                    exit_rec.slippage_est,
                    exit_rec.notes,
                ),
            )
            self._conn.commit()
        except Exception as exc:
            log.error("trade_log: persist_exit error -- %s", exc)

    def _query_closed_trades(
        self,
        since:  Optional[datetime],
        symbol: Optional[str],
        limit:  int,
    ) -> list[tuple[TradeEntry, TradeExit]]:
        if self._conn is None:
            return []
        try:
            conditions = []
            params: list[Any] = []
            if since:
                conditions.append("e.ts_entry >= ?")
                params.append(since.isoformat())
            if symbol:
                conditions.append("e.symbol = ?")
                params.append(symbol.upper())

            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            sql = f"""
                SELECT e.*, x.ts_exit, x.exit_price, x.qty_closed, x.reason,
                       x.pnl_abs, x.pnl_pct, x.bars_held,
                       x.bh_mass_at_exit, x.tf_score_at_exit, x.rl_exit_signal,
                       x.slippage_est, x.notes
                FROM trade_entries e
                JOIN trade_exits x ON x.trade_id = e.trade_id
                {where}
                ORDER BY e.ts_entry DESC
                LIMIT ?
            """
            params.append(limit)
            cur  = self._conn.execute(sql, params)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            result = []
            for row in rows:
                d = dict(zip(cols, row))
                try:
                    entry, exit_rec = self._row_to_objects(d)
                    result.append((entry, exit_rec))
                except Exception as exc:
                    log.debug("trade_log: row parse error -- %s", exc)
            return result
        except Exception as exc:
            log.error("trade_log: query_closed_trades error -- %s", exc)
            return []

    def _fetch_entry_context_from_db(self, trade_id: str) -> dict[str, Any]:
        if self._conn is None:
            return {}
        try:
            cur = self._conn.execute(
                "SELECT * FROM trade_entries WHERE trade_id = ?", (trade_id,)
            )
            row = cur.fetchone()
            if not row:
                return {}
            cols = [d[0] for d in cur.description]
            d    = dict(zip(cols, row))
            if d.get("param_snapshot"):
                try:
                    d["param_snapshot"] = json.loads(d["param_snapshot"])
                except Exception:
                    pass
            if d.get("context_json"):
                try:
                    d["context"] = json.loads(d["context_json"])
                except Exception:
                    pass
            return d
        except Exception as exc:
            log.error("trade_log: fetch_entry_context error -- %s", exc)
            return {}

    @staticmethod
    def _row_to_objects(d: dict[str, Any]) -> tuple[TradeEntry, TradeExit]:
        """Convert a joined DB row dict to (TradeEntry, TradeExit)."""
        param_snap = {}
        if d.get("param_snapshot"):
            try:
                param_snap = json.loads(d["param_snapshot"])
            except Exception:
                pass
        ctx = {}
        if d.get("context_json"):
            try:
                ctx = json.loads(d["context_json"])
            except Exception:
                pass

        entry = TradeEntry(
            trade_id            = d["trade_id"],
            ts_entry            = datetime.fromisoformat(d["ts_entry"]),
            symbol              = d["symbol"],
            side                = d["side"],
            entry_price         = d["entry_price"],
            qty                 = d["qty"],
            notional            = d.get("notional") or 0.0,
            bh_mass_15m         = d.get("bh_mass_15m"),
            bh_mass_1h          = d.get("bh_mass_1h"),
            bh_mass_4h          = d.get("bh_mass_4h"),
            bh_dir_15m          = d.get("bh_dir_15m"),
            bh_dir_1h           = d.get("bh_dir_1h"),
            bh_dir_4h           = d.get("bh_dir_4h"),
            cf_bull             = d.get("cf_bull"),
            cf_bear             = d.get("cf_bear"),
            cf_score            = d.get("cf_score"),
            tf_score            = d.get("tf_score"),
            garch_vol           = d.get("garch_vol"),
            garch_vol_scale     = d.get("garch_vol_scale"),
            hurst_exp           = d.get("hurst_exp"),
            ou_zscore           = d.get("ou_zscore"),
            ml_signal           = d.get("ml_signal"),
            granger_boost       = d.get("granger_boost"),
            rl_confidence       = d.get("rl_confidence"),
            nav_omega           = d.get("nav_omega"),
            nav_omega_ratio     = d.get("nav_omega_ratio"),
            nav_geodesic        = d.get("nav_geodesic"),
            nav_geodesic_ratio  = d.get("nav_geodesic_ratio"),
            regime              = d.get("regime"),
            corr_mode           = d.get("corr_mode"),
            dd_from_peak        = d.get("dd_from_peak"),
            event_blocked       = bool(d.get("event_blocked", 0)),
            portfolio_nav       = d.get("portfolio_nav"),
            crypto_exposure     = d.get("crypto_exposure"),
            equity_exposure     = d.get("equity_exposure"),
            param_snapshot      = param_snap,
            context             = ctx,
        )

        exit_rec = TradeExit(
            trade_id         = d["trade_id"],
            ts_exit          = datetime.fromisoformat(d["ts_exit"]),
            exit_price       = d["exit_price"],
            qty_closed       = d["qty_closed"],
            reason           = d["reason"],
            pnl_abs          = d.get("pnl_abs"),
            pnl_pct          = d.get("pnl_pct"),
            bars_held        = d.get("bars_held"),
            bh_mass_at_exit  = d.get("bh_mass_at_exit"),
            tf_score_at_exit = d.get("tf_score_at_exit"),
            rl_exit_signal   = d.get("rl_exit_signal"),
            slippage_est     = d.get("slippage_est"),
            notes            = d.get("notes") or "",
        )
        return entry, exit_rec

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _opt_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _opt_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_singleton: Optional[TradeLog] = None
_singleton_lock = threading.Lock()


def get_trade_log(db_path: Optional[Path] = None) -> TradeLog:
    """Return (or create) the module-level TradeLog singleton."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = TradeLog(db_path=db_path)
    return _singleton


def reset_singleton_for_testing(db_path: Optional[Path] = None) -> TradeLog:
    """Force-create a new singleton. For use in tests only."""
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            _singleton.close()
        _singleton = TradeLog(db_path=db_path)
    return _singleton
