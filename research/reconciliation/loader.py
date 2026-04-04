"""
research/reconciliation/loader.py
===================================
Data ingestion and normalisation layer for the live-vs-backtest reconciliation
pipeline.

Classes
-------
TradeRecord         – canonical dataclass for a single closed trade
LiveTradeLoader     – reads from the live_trades SQLite database
BacktestTradeLoader – reads from the crypto_trades CSV (or a backtest SQLite)

Functions
---------
merge_live_backtest(live, backtest) → pd.DataFrame
    Align live and backtest trades on (sym, approximate_entry_time) and
    produce a unified comparison DataFrame.

Notes
-----
* All timestamps are stored as UTC-aware pandas Timestamps internally.
* The live DB schema may evolve; the loader handles several schema variants
  by probing column names at runtime.
* Derived fields (return_pct, hold_hours, dollar_return_pct) are computed
  after loading so that both live and backtest records expose the same
  feature surface.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import warnings
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

REGIME_VALUES = frozenset({"BULL", "BEAR", "SIDEWAYS", "HIGH_VOL", "UNKNOWN"})

# Columns we expect in the backtest CSV (minimum required set)
BT_CSV_REQUIRED = {"exit_time", "sym", "entry_price", "exit_price", "dollar_pos", "pnl", "hold_bars"}

# Live DB table name priority order
LIVE_TABLE_CANDIDATES = ["trades", "live_trades", "fills"]

# ── TradeRecord ──────────────────────────────────────────────────────────────


@dataclass
class TradeRecord:
    """
    Canonical representation of one closed trade, covering both live and
    backtest sources.

    Fields marked Optional[...] may be None when the source does not provide
    them; derived fields are always populated after normalisation.
    """

    # --- identity ---
    sym: str
    source: str  # "live" | "backtest"
    trade_id: Optional[str] = None

    # --- timing ---
    entry_time: Optional[pd.Timestamp] = None
    exit_time: Optional[pd.Timestamp] = None

    # --- pricing ---
    entry_price: float = float("nan")
    exit_price: float = float("nan")

    # --- sizing ---
    dollar_pos: float = float("nan")       # notional position in USD
    qty: float = float("nan")              # number of units / contracts

    # --- outcome ---
    pnl: float = float("nan")
    hold_bars: int = 0

    # --- regime & signal context ---
    regime: str = "UNKNOWN"
    tf_score: float = float("nan")         # BH timeframe confluence score
    mass: float = float("nan")             # BH mass (momentum proxy)
    atr: float = float("nan")             # ATR at entry
    delta_score: float = float("nan")      # tf_score × mass × ATR
    ensemble_signal: float = float("nan")  # aggregated D3QN/DDQN/TD3QN signal

    # --- exit metadata ---
    exit_reason: str = "UNKNOWN"
    side: str = "LONG"  # "LONG" | "SHORT"

    # --- derived (computed by normalise()) ---
    return_pct: float = float("nan")
    hold_hours: float = float("nan")
    dollar_return_pct: float = float("nan")  # pnl / dollar_pos * 100

    # ── helpers ──────────────────────────────────────────────────────────────

    def normalise(self) -> "TradeRecord":
        """
        Compute derived fields in-place. Call after all raw fields are set.
        Returns self for chaining.
        """
        # return_pct from prices
        if not (np.isnan(self.entry_price) or np.isnan(self.exit_price) or self.entry_price == 0):
            sign = 1.0 if self.side == "LONG" else -1.0
            self.return_pct = sign * (self.exit_price - self.entry_price) / self.entry_price * 100.0
        elif not np.isnan(self.pnl) and not np.isnan(self.dollar_pos) and self.dollar_pos != 0:
            self.return_pct = self.pnl / self.dollar_pos * 100.0

        # dollar_return_pct
        if not (np.isnan(self.pnl) or np.isnan(self.dollar_pos)) and self.dollar_pos != 0:
            self.dollar_return_pct = self.pnl / self.dollar_pos * 100.0

        # hold_hours from timestamps
        if self.entry_time is not None and self.exit_time is not None:
            delta = self.exit_time - self.entry_time
            self.hold_hours = delta.total_seconds() / 3600.0
        elif self.hold_bars > 0:
            # Assume 1-hour bars as default; callers can override
            self.hold_hours = float(self.hold_bars)

        # delta_score from components
        if np.isnan(self.delta_score):
            parts = [self.tf_score, self.mass, self.atr]
            if not any(np.isnan(p) for p in parts):
                self.delta_score = self.tf_score * self.mass * self.atr

        return self

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # convert timestamps to ISO strings for serialisation
        for k in ("entry_time", "exit_time"):
            if d[k] is not None:
                d[k] = str(d[k])
        return d

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


# ── Utility functions ────────────────────────────────────────────────────────


def _to_utc(ts: Any) -> Optional[pd.Timestamp]:
    """
    Convert anything timestamp-like to a UTC-aware pd.Timestamp.
    Returns None on failure.
    """
    if ts is None:
        return None
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        return t
    except Exception:
        return None


def _infer_regime(row: pd.Series) -> str:
    """
    Infer market regime from available signal columns.
    Uses a simple heuristic based on garch_vol and trend proxy.
    """
    garch_vol = row.get("garch_vol", np.nan)
    tf_score = row.get("tf_score", 0) or 0
    ou_zscore = row.get("ou_zscore", np.nan)

    if not np.isnan(garch_vol):
        if garch_vol > 0.04:
            return "HIGH_VOL"
    if tf_score >= 2:
        return "BULL"
    elif tf_score <= -2:
        return "BEAR"
    if not np.isnan(ou_zscore):
        if abs(ou_zscore) < 1.0:
            return "SIDEWAYS"
    return "SIDEWAYS"


def _safe_float(val: Any, default: float = float("nan")) -> float:
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _records_to_df(records: list[TradeRecord]) -> pd.DataFrame:
    """Convert a list of TradeRecord objects to a tidy DataFrame."""
    if not records:
        return pd.DataFrame(columns=TradeRecord.field_names())
    rows = [r.to_dict() for r in records]
    df = pd.DataFrame(rows)
    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


# ── LiveTradeLoader ──────────────────────────────────────────────────────────


class LiveTradeLoader:
    """
    Loads closed trades from the live_trades SQLite database produced by
    live_trader_alpaca.py.

    Schema variants handled
    -----------------------
    Version A (current): trades table with columns
        id, ts, symbol, side, qty, price, entry_price, pnl, bars_held,
        equity_after, trade_duration_s, notes
    Version B (legacy): trades table may have 'exit_price' instead of 'price'
    Version C: may include a 'regime' text column
    Version D: separate regime_log table (joined on ts, symbol)

    Parameters
    ----------
    db_path : str | Path
        Path to live_trades.db
    bar_duration_hours : float
        Duration of one bar in hours, used to estimate hold_hours when
        timestamp data is unavailable (default 1.0 for 1-hour bars).
    """

    def __init__(
        self,
        db_path: str | Path,
        bar_duration_hours: float = 1.0,
    ) -> None:
        self.db_path = Path(db_path)
        self.bar_duration_hours = bar_duration_hours
        self._conn: Optional[sqlite3.Connection] = None

    # ── connection management ─────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        if not self.db_path.exists():
            raise FileNotFoundError(f"Live DB not found: {self.db_path}")
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _available_tables(self, conn: sqlite3.Connection) -> list[str]:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [r[0] for r in cur.fetchall()]

    def _available_columns(self, conn: sqlite3.Connection, table: str) -> list[str]:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return [r["name"] for r in cur.fetchall()]

    # ── regime_log join ───────────────────────────────────────────────────

    def _load_regime_log(self, conn: sqlite3.Connection) -> pd.DataFrame:
        """
        Load regime_log if present and return as a DataFrame indexed by
        (ts, symbol) for merging.
        """
        tables = self._available_tables(conn)
        if "regime_log" not in tables:
            return pd.DataFrame()
        try:
            df = pd.read_sql("SELECT * FROM regime_log", conn)
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            return df
        except Exception as exc:
            log.warning("Could not load regime_log: %s", exc)
            return pd.DataFrame()

    # ── raw query ─────────────────────────────────────────────────────────

    def _raw_trades(self, conn: sqlite3.Connection) -> pd.DataFrame:
        """
        Detect the trades table and select all rows, adapting to schema
        variants.
        """
        tables = self._available_tables(conn)

        # Find the right table
        trade_table: Optional[str] = None
        for candidate in LIVE_TABLE_CANDIDATES:
            if candidate in tables:
                trade_table = candidate
                break
        if trade_table is None:
            raise ValueError(
                f"No trades table found in {self.db_path}. "
                f"Available: {tables}"
            )

        cols = self._available_columns(conn, trade_table)
        col_set = set(cols)

        # Build SELECT list
        select_parts = ["*"]
        extra_exprs: list[str] = []

        # Normalise exit_price / price
        if "exit_price" not in col_set and "price" in col_set:
            extra_exprs.append("price AS exit_price")
        if "bars_held" in col_set and "hold_bars" not in col_set:
            extra_exprs.append("bars_held AS hold_bars")
        if "symbol" in col_set and "sym" not in col_set:
            extra_exprs.append("symbol AS sym")

        if extra_exprs:
            select_parts = ["*"] + extra_exprs

        query = f"SELECT {', '.join(select_parts)} FROM {trade_table}"
        df = pd.read_sql(query, conn)

        # Parse timestamp
        for ts_col in ("ts", "timestamp", "exit_time", "datetime"):
            if ts_col in df.columns:
                df["_ts_raw"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
                break
        else:
            df["_ts_raw"] = pd.NaT

        return df

    # ── normalise ─────────────────────────────────────────────────────────

    def _normalise_row(
        self,
        row: pd.Series,
        regime_map: dict[tuple, pd.Series],
    ) -> TradeRecord:
        sym = str(row.get("sym") or row.get("symbol") or "UNKNOWN")
        ts_exit = _to_utc(row.get("_ts_raw") or row.get("ts"))

        # Estimate entry time from hold_bars
        hold_bars = int(row.get("hold_bars", 0) or row.get("bars_held", 0) or 0)
        if ts_exit is not None and hold_bars > 0:
            ts_entry = ts_exit - pd.Timedelta(hours=hold_bars * self.bar_duration_hours)
        else:
            ts_entry = None

        exit_price = _safe_float(row.get("exit_price") or row.get("price"))
        entry_price = _safe_float(row.get("entry_price"))
        qty = _safe_float(row.get("qty"))
        pnl = _safe_float(row.get("pnl"))

        # Compute dollar_pos: qty * entry_price if not stored
        dollar_pos = _safe_float(row.get("dollar_pos"))
        if np.isnan(dollar_pos) and not (np.isnan(qty) or np.isnan(entry_price)):
            dollar_pos = abs(qty * entry_price)

        side = str(row.get("side") or "LONG").upper()
        notes = str(row.get("notes") or "")
        exit_reason = "UNKNOWN"
        if notes:
            for keyword in ("STOP", "TARGET", "SIGNAL", "TIMEOUT", "TRAILING"):
                if keyword in notes.upper():
                    exit_reason = keyword
                    break

        # Try to get regime signal data from regime_log
        regime_row = regime_map.get((ts_exit, sym), pd.Series(dtype=float))
        tf_score = _safe_float(row.get("tf_score") or regime_row.get("tf_score"))
        atr = _safe_float(row.get("atr") or regime_row.get("atr"))
        delta_score = _safe_float(row.get("delta_score") or regime_row.get("delta_score"))
        garch_vol = _safe_float(regime_row.get("garch_vol"))
        ou_zscore = _safe_float(regime_row.get("ou_zscore"))

        # Compute mass from BH physics: delta_score = tf_score * mass * atr
        mass = float("nan")
        if not (np.isnan(delta_score) or np.isnan(tf_score) or np.isnan(atr)):
            if tf_score != 0 and atr != 0:
                mass = delta_score / (tf_score * atr)

        # Infer regime
        regime_proxy = pd.Series(
            {"tf_score": tf_score, "garch_vol": garch_vol, "ou_zscore": ou_zscore}
        )
        regime = _infer_regime(regime_proxy)

        rec = TradeRecord(
            sym=sym,
            source="live",
            trade_id=str(row.get("id") or ""),
            entry_time=ts_entry,
            exit_time=ts_exit,
            entry_price=entry_price,
            exit_price=exit_price,
            dollar_pos=dollar_pos,
            qty=qty,
            pnl=pnl,
            hold_bars=hold_bars,
            regime=regime,
            tf_score=tf_score,
            mass=mass,
            atr=atr,
            delta_score=delta_score,
            ensemble_signal=float("nan"),  # not stored in live DB
            exit_reason=exit_reason,
            side=side,
        )
        return rec.normalise()

    # ── public API ────────────────────────────────────────────────────────

    def load(self) -> list[TradeRecord]:
        """
        Load all closed trades from the SQLite database.

        Returns
        -------
        list[TradeRecord]
            Normalised trade records sorted by exit_time ascending.
        """
        conn = self._connect()
        try:
            regime_df = self._load_regime_log(conn)
            # Build lookup map (ts, symbol) → Series for fast join
            regime_map: dict[tuple, pd.Series] = {}
            if not regime_df.empty and "ts" in regime_df.columns and "symbol" in regime_df.columns:
                for _, rrow in regime_df.iterrows():
                    key = (_to_utc(rrow.get("ts")), str(rrow.get("symbol", "")))
                    regime_map[key] = rrow

            raw = self._raw_trades(conn)
            if raw.empty:
                log.info("live_trades.db contains no trades yet.")
                return []

            records: list[TradeRecord] = []
            for _, row in raw.iterrows():
                try:
                    rec = self._normalise_row(row, regime_map)
                    records.append(rec)
                except Exception as exc:
                    log.warning("Skipping malformed live trade row: %s", exc)

            # Sort by exit_time, putting None at end
            records.sort(
                key=lambda r: (r.exit_time is None, r.exit_time or pd.Timestamp.min.tz_localize("UTC"))
            )
            log.info("Loaded %d live trades.", len(records))
            return records

        finally:
            conn.close()

    def load_df(self) -> pd.DataFrame:
        """Load and return as a DataFrame."""
        return _records_to_df(self.load())

    def load_equity_snapshots(self) -> pd.DataFrame:
        """
        Load equity curve snapshots if available.

        Returns
        -------
        pd.DataFrame
            Columns: ts (UTC Timestamp), equity, positions (JSON str)
        """
        conn = self._connect()
        try:
            tables = self._available_tables(conn)
            if "equity_snapshots" not in tables:
                return pd.DataFrame(columns=["ts", "equity", "positions"])
            df = pd.read_sql("SELECT * FROM equity_snapshots ORDER BY ts", conn)
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            return df
        finally:
            conn.close()

    def summary_stats(self) -> dict[str, Any]:
        """
        Return quick summary statistics without building full TradeRecord
        objects.
        """
        records = self.load()
        if not records:
            return {"count": 0}
        pnls = [r.pnl for r in records if not np.isnan(r.pnl)]
        return {
            "count": len(records),
            "total_pnl": float(np.nansum(pnls)),
            "win_rate": float(np.mean([p > 0 for p in pnls])) if pnls else float("nan"),
            "avg_pnl": float(np.nanmean(pnls)) if pnls else float("nan"),
            "syms": list({r.sym for r in records}),
        }


# ── BacktestTradeLoader ──────────────────────────────────────────────────────


class BacktestTradeLoader:
    """
    Loads backtest trades from:
      - A CSV file matching the crypto_trades.csv format
        (exit_time, sym, entry_price, exit_price, dollar_pos, pnl, hold_bars)
      - A SQLite backtest database with a compatible schema

    Parameters
    ----------
    path : str | Path
        Path to the CSV or SQLite file.
    bar_duration_hours : float
        Hours per bar, used to reconstruct entry_time (default 1.0).
    regime_csv : str | Path | None
        Optional path to a separate regime classification CSV with columns
        (exit_time, sym, regime, tf_score, mass, atr, ensemble_signal).
    """

    def __init__(
        self,
        path: str | Path,
        bar_duration_hours: float = 1.0,
        regime_csv: Optional[str | Path] = None,
    ) -> None:
        self.path = Path(path)
        self.bar_duration_hours = bar_duration_hours
        self.regime_csv = Path(regime_csv) if regime_csv else None

    # ── internals ─────────────────────────────────────────────────────────

    def _load_regime_supplement(self) -> pd.DataFrame:
        if self.regime_csv is None or not self.regime_csv.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.regime_csv)
            df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
            return df
        except Exception as exc:
            log.warning("Could not load regime supplement CSV: %s", exc)
            return pd.DataFrame()

    def _load_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        missing = BT_CSV_REQUIRED - set(df.columns)
        if missing:
            raise ValueError(
                f"Backtest CSV missing required columns: {missing}. "
                f"Found: {list(df.columns)}"
            )
        df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
        return df

    def _load_sqlite(self) -> pd.DataFrame:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            for candidate in ["backtest_trades", "trades", "bt_trades"]:
                if candidate in tables:
                    df = pd.read_sql(f"SELECT * FROM {candidate}", conn)
                    # Column renaming to match CSV spec
                    rename_map = {
                        "symbol": "sym",
                        "exit_timestamp": "exit_time",
                        "position_size": "dollar_pos",
                        "bars": "hold_bars",
                    }
                    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
                    if "exit_time" in df.columns:
                        df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
                    return df
            raise ValueError(f"No recognisable backtest table in {self.path}. Available: {tables}")
        finally:
            conn.close()

    def _detect_format(self) -> str:
        suffix = self.path.suffix.lower()
        if suffix in (".csv", ".tsv", ".txt"):
            return "csv"
        if suffix in (".db", ".sqlite", ".sqlite3"):
            return "sqlite"
        # Try to sniff
        try:
            with open(self.path, "rb") as fh:
                magic = fh.read(16)
            if magic.startswith(b"SQLite format 3"):
                return "sqlite"
        except Exception:
            pass
        return "csv"

    def _normalise_row(
        self,
        row: pd.Series,
        regime_map: dict[tuple, pd.Series],
    ) -> TradeRecord:
        sym = str(row.get("sym") or row.get("symbol") or "UNKNOWN")
        exit_time = _to_utc(row.get("exit_time"))
        hold_bars = int(_safe_float(row.get("hold_bars", 0)) or 0)

        entry_price = _safe_float(row.get("entry_price"))
        exit_price = _safe_float(row.get("exit_price"))
        dollar_pos = _safe_float(row.get("dollar_pos"))
        pnl = _safe_float(row.get("pnl"))

        if exit_time is not None and hold_bars > 0:
            entry_time = exit_time - pd.Timedelta(hours=hold_bars * self.bar_duration_hours)
        else:
            entry_time = None

        regime_row = regime_map.get((exit_time, sym), pd.Series(dtype=float))
        regime = str(regime_row.get("regime", "UNKNOWN") or "UNKNOWN")
        if regime not in REGIME_VALUES:
            regime = "UNKNOWN"
        tf_score = _safe_float(row.get("tf_score") or regime_row.get("tf_score"))
        mass = _safe_float(row.get("mass") or regime_row.get("mass"))
        atr = _safe_float(row.get("atr") or regime_row.get("atr"))
        ensemble_signal = _safe_float(
            row.get("ensemble_signal") or regime_row.get("ensemble_signal")
        )

        delta_score = float("nan")
        if not any(np.isnan(v) for v in [tf_score, mass, atr]):
            delta_score = tf_score * mass * atr

        # Infer regime if not provided
        if regime == "UNKNOWN":
            regime = _infer_regime(pd.Series({"tf_score": tf_score}))

        exit_reason = str(row.get("exit_reason") or "UNKNOWN")

        rec = TradeRecord(
            sym=sym,
            source="backtest",
            trade_id=str(row.get("id") or row.get("trade_id") or ""),
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            dollar_pos=dollar_pos,
            qty=float("nan"),
            pnl=pnl,
            hold_bars=hold_bars,
            regime=regime,
            tf_score=tf_score,
            mass=mass,
            atr=atr,
            delta_score=delta_score,
            ensemble_signal=ensemble_signal,
            exit_reason=exit_reason,
            side="LONG",  # backtest is long-only by default
        )
        return rec.normalise()

    # ── public API ────────────────────────────────────────────────────────

    def load(self) -> list[TradeRecord]:
        """
        Load all backtest trades and return as normalised TradeRecord list.
        """
        fmt = self._detect_format()
        if fmt == "csv":
            raw = self._load_csv()
        else:
            raw = self._load_sqlite()

        regime_df = self._load_regime_supplement()
        regime_map: dict[tuple, pd.Series] = {}
        if not regime_df.empty:
            for _, rrow in regime_df.iterrows():
                key = (_to_utc(rrow.get("exit_time")), str(rrow.get("sym", "")))
                regime_map[key] = rrow

        records: list[TradeRecord] = []
        for _, row in raw.iterrows():
            try:
                rec = self._normalise_row(row, regime_map)
                records.append(rec)
            except Exception as exc:
                log.warning("Skipping malformed backtest row: %s", exc)

        records.sort(
            key=lambda r: (r.exit_time is None, r.exit_time or pd.Timestamp.min.tz_localize("UTC"))
        )
        log.info("Loaded %d backtest trades.", len(records))
        return records

    def load_df(self) -> pd.DataFrame:
        return _records_to_df(self.load())

    def summary_stats(self) -> dict[str, Any]:
        records = self.load()
        if not records:
            return {"count": 0}
        pnls = [r.pnl for r in records if not np.isnan(r.pnl)]
        returns = [r.return_pct for r in records if not np.isnan(r.return_pct)]
        wins = [p > 0 for p in pnls]
        losses = [p for p in pnls if p <= 0]
        gain = [p for p in pnls if p > 0]
        profit_factor = (
            abs(sum(gain)) / abs(sum(losses))
            if losses and abs(sum(losses)) > 0
            else float("inf")
        )
        return {
            "count": len(records),
            "total_pnl": float(np.nansum(pnls)),
            "win_rate": float(np.mean(wins)) if wins else float("nan"),
            "avg_pnl": float(np.nanmean(pnls)) if pnls else float("nan"),
            "avg_return_pct": float(np.nanmean(returns)) if returns else float("nan"),
            "profit_factor": profit_factor,
            "syms": list({r.sym for r in records}),
        }


# ── merge_live_backtest ───────────────────────────────────────────────────────


def merge_live_backtest(
    live: list[TradeRecord] | pd.DataFrame,
    backtest: list[TradeRecord] | pd.DataFrame,
    time_tolerance_hours: float = 2.0,
    match_on_sym: bool = True,
) -> pd.DataFrame:
    """
    Align live and backtest trades to produce a side-by-side comparison
    DataFrame.

    Matching strategy
    -----------------
    For each live trade, find the closest backtest trade for the same symbol
    (if ``match_on_sym`` is True) within ``time_tolerance_hours``.  If no
    match is found the live trade is included with NaN backtest columns.

    Unmatched backtest trades are appended with NaN live columns.

    Parameters
    ----------
    live, backtest : list[TradeRecord] | pd.DataFrame
        Source data from the loaders.
    time_tolerance_hours : float
        Maximum exit-time difference for considering two trades as
        "the same" signal firing.
    match_on_sym : bool
        Require the same symbol to match (default True).

    Returns
    -------
    pd.DataFrame
        Columns: all TradeRecord fields prefixed with ``live_`` or ``bt_``,
        plus ``time_diff_hours``, ``pnl_diff``, ``return_diff_pct``.
    """
    # Convert to DataFrames if needed
    if isinstance(live, list):
        live_df = _records_to_df(live)
    else:
        live_df = live.copy()

    if isinstance(backtest, list):
        bt_df = _records_to_df(backtest)
    else:
        bt_df = backtest.copy()

    # Work on copies with standardised exit_time
    for df, label in ((live_df, "live"), (bt_df, "backtest")):
        if "exit_time" not in df.columns:
            df["exit_time"] = pd.NaT
        df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
        if "sym" not in df.columns:
            df["sym"] = "UNKNOWN"

    tol = pd.Timedelta(hours=time_tolerance_hours)

    matched_pairs: list[dict[str, Any]] = []
    bt_used: set[int] = set()

    for live_idx, lrow in live_df.iterrows():
        l_sym = lrow["sym"]
        l_time = lrow["exit_time"]

        if match_on_sym:
            candidates = bt_df[bt_df["sym"] == l_sym]
        else:
            candidates = bt_df

        if l_time is pd.NaT or candidates.empty:
            # No match possible
            pair = _build_merge_row(lrow, None, live_df.columns, bt_df.columns)
            matched_pairs.append(pair)
            continue

        time_diffs = (candidates["exit_time"] - l_time).abs()
        valid = time_diffs[time_diffs <= tol]

        if valid.empty:
            pair = _build_merge_row(lrow, None, live_df.columns, bt_df.columns)
        else:
            best_idx = valid.idxmin()
            if best_idx in bt_used:
                pair = _build_merge_row(lrow, None, live_df.columns, bt_df.columns)
            else:
                bt_row = bt_df.loc[best_idx]
                bt_used.add(best_idx)
                pair = _build_merge_row(lrow, bt_row, live_df.columns, bt_df.columns)

        matched_pairs.append(pair)

    # Append unmatched backtest rows
    for bt_idx, btrow in bt_df.iterrows():
        if bt_idx not in bt_used:
            pair = _build_merge_row(None, btrow, live_df.columns, bt_df.columns)
            matched_pairs.append(pair)

    result = pd.DataFrame(matched_pairs)

    # Compute diff columns
    for col_suffix, l_col, b_col in [
        ("pnl_diff", "live_pnl", "bt_pnl"),
        ("return_diff_pct", "live_return_pct", "bt_return_pct"),
    ]:
        if l_col in result.columns and b_col in result.columns:
            result[col_suffix] = pd.to_numeric(result[l_col], errors="coerce") - pd.to_numeric(result[b_col], errors="coerce")

    if "live_exit_time" in result.columns and "bt_exit_time" in result.columns:
        lt = pd.to_datetime(result["live_exit_time"], utc=True, errors="coerce")
        bt = pd.to_datetime(result["bt_exit_time"], utc=True, errors="coerce")
        result["time_diff_hours"] = (lt - bt).dt.total_seconds() / 3600.0

    return result


def _build_merge_row(
    live_row: Optional[pd.Series],
    bt_row: Optional[pd.Series],
    live_cols: Sequence[str],
    bt_cols: Sequence[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {}

    for col in live_cols:
        row[f"live_{col}"] = live_row[col] if live_row is not None else None

    for col in bt_cols:
        row[f"bt_{col}"] = bt_row[col] if bt_row is not None else None

    return row


# ── Rolling performance helpers ───────────────────────────────────────────────


def compute_rolling_metrics(
    df: pd.DataFrame,
    pnl_col: str = "pnl",
    window: int = 50,
    annual_factor: float = 252.0,
) -> pd.DataFrame:
    """
    Compute rolling Sharpe ratio, win rate, and drawdown for a trade
    DataFrame sorted by exit time.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least the pnl_col column.
    pnl_col : str
        Column name for per-trade PnL.
    window : int
        Rolling window in number of trades.
    annual_factor : float
        Annualisation factor (default 252 trading days).

    Returns
    -------
    pd.DataFrame
        Same index as ``df`` with added columns:
        rolling_sharpe, rolling_win_rate, rolling_max_dd, cum_pnl.
    """
    out = df.copy()
    pnl = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)
    out["cum_pnl"] = pnl.cumsum()

    roll_mean = pnl.rolling(window, min_periods=max(2, window // 5)).mean()
    roll_std = pnl.rolling(window, min_periods=max(2, window // 5)).std()
    out["rolling_sharpe"] = roll_mean / roll_std.replace(0, np.nan) * np.sqrt(annual_factor)
    out["rolling_win_rate"] = (pnl > 0).rolling(window, min_periods=1).mean()

    cum = out["cum_pnl"]
    running_max = cum.cummax()
    out["rolling_max_dd"] = (cum - running_max)

    return out


def stratify_by_regime(
    df: pd.DataFrame,
    regime_col: str = "regime",
) -> dict[str, pd.DataFrame]:
    """
    Split a trade DataFrame into sub-DataFrames by regime label.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are regime strings; values are the filtered DataFrames.
    """
    if regime_col not in df.columns:
        return {"ALL": df}
    result: dict[str, pd.DataFrame] = {}
    for regime in df[regime_col].unique():
        result[str(regime)] = df[df[regime_col] == regime].copy()
    return result


def align_to_calendar(
    live_df: pd.DataFrame,
    bt_df: pd.DataFrame,
    freq: str = "D",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Resample both DataFrames to a common calendar grid so they can be
    compared period-by-period.

    Parameters
    ----------
    live_df, bt_df : pd.DataFrame
        Trade DataFrames with ``exit_time`` column.
    freq : str
        Pandas frequency string ('D', 'W', 'ME', etc.)

    Returns
    -------
    (live_resampled, bt_resampled) : tuple[pd.DataFrame, pd.DataFrame]
        Each has columns: period_start, pnl_sum, trade_count, win_rate.
    """
    def _resample(df: pd.DataFrame, label: str) -> pd.DataFrame:
        if df.empty or "exit_time" not in df.columns:
            return pd.DataFrame(columns=["period_start", "pnl_sum", "trade_count", "win_rate"])
        ts = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
        pnl = pd.to_numeric(df.get("pnl", pd.Series(dtype=float)), errors="coerce")
        tmp = pd.DataFrame({"ts": ts, "pnl": pnl})
        tmp = tmp.dropna(subset=["ts"]).set_index("ts").sort_index()
        agg = tmp.resample(freq).agg(
            pnl_sum=("pnl", "sum"),
            trade_count=("pnl", "count"),
            win_rate=("pnl", lambda x: (x > 0).mean()),
        )
        agg["source"] = label
        agg.index.name = "period_start"
        return agg.reset_index()

    return _resample(live_df, "live"), _resample(bt_df, "backtest")
