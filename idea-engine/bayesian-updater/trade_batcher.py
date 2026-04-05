"""
trade_batcher.py
================
Load, validate, and batch live trades from the SQLite database at
``tools/backtest_output/live_trades.db``.

The batcher supports two modes:

1. **Single-batch mode** -- aggregate all trades into one TradeStats.
2. **A/B split mode** -- group trades by ``experiment_tag`` column (if
   present) so that parameter-specific posteriors can be updated
   independently for each experimental arm.

TradeStats is the core result object consumed by the posterior updater
and the likelihood functions.

Database schema (expected)
--------------------------
    trades (
        id              INTEGER PRIMARY KEY,
        timestamp       TEXT,       -- ISO-8601 UTC
        symbol          TEXT,
        entry_price     REAL,
        exit_price      REAL,
        pnl_frac        REAL,       -- fractional P&L (exit/entry - 1)
        hold_bars       INTEGER,    -- number of 15-minute bars held
        entry_hour      INTEGER,    -- UTC hour of entry (0-23)
        signal_hour     INTEGER,    -- UTC hour signal was generated
        was_entered     INTEGER,    -- 1 if signal passed staleness check
        was_protected   INTEGER,    -- 1 if winner-protection exit triggered
        was_winner      INTEGER,    -- 1 if trade was profitable at peak
        experiment_tag  TEXT,       -- optional A/B label
        param_snapshot  TEXT        -- JSON of params used for this trade
    )

If columns are missing (older schema), the batcher degrades gracefully
and sets missing statistics to 0.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Resolve DB path relative to this file's location in the repo
_HERE  = Path(__file__).resolve()
_REPO  = _HERE.parents[3]  # srfm-lab/
_DEFAULT_DB = _REPO / "tools" / "backtest_output" / "live_trades.db"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """
    Lightweight representation of a single live trade record.

    All attributes default to sensible nulls so that incomplete rows do
    not crash downstream code.
    """

    id:             int   = 0
    timestamp:      str   = ""
    symbol:         str   = "UNKNOWN"
    entry_price:    float = 0.0
    exit_price:     float = 0.0
    pnl_frac:       float = 0.0
    hold_bars:      int   = 0
    entry_hour:     int   = 0
    signal_hour:    int   = 0
    was_entered:    int   = 1   # default: assume entered
    was_protected:  int   = 0
    was_winner:     int   = 0
    experiment_tag: str   = "default"
    param_snapshot: dict  = field(default_factory=dict)

    @property
    def is_win(self) -> bool:
        return self.pnl_frac > 0.0


@dataclass
class TradeStats:
    """
    Aggregated statistics for a batch of trades.

    These are the sufficient statistics consumed by the likelihood
    functions in likelihood.py.

    Attributes
    ----------
    n_trades          : total number of trades.
    n_wins            : number of profitable trades (pnl_frac > 0).
    win_rate          : n_wins / n_trades.
    avg_pnl           : mean fractional P&L.
    avg_hold          : mean hold duration in bars.
    pnl_list          : list of per-trade pnl_frac values.
    boosted_hour_pnl  : P&L for trades entered during "boost hours" (7-11, 14-17 UTC).
    n_entered         : signals that passed the staleness check.
    n_total_signals   : all signals generated (entered + rejected).
    n_protected_wins  : winners that triggered winner-protection exit.
    n_winners         : total profitable trades at peak (was_winner=1).
    experiment_tag    : A/B label this batch belongs to.
    """

    n_trades:         int   = 0
    n_wins:           int   = 0
    win_rate:         float = 0.0
    avg_pnl:          float = 0.0
    avg_hold:         float = 0.0
    pnl_list:         List[float] = field(default_factory=list)
    boosted_hour_pnl: List[float] = field(default_factory=list)
    n_entered:        int   = 0
    n_total_signals:  int   = 0
    n_protected_wins: int   = 0
    n_winners:        int   = 0
    experiment_tag:   str   = "default"

    def is_empty(self) -> bool:
        return self.n_trades == 0

    def to_dict(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "n_wins": self.n_wins,
            "win_rate": self.win_rate,
            "avg_pnl": self.avg_pnl,
            "avg_hold": self.avg_hold,
            "n_entered": self.n_entered,
            "n_total_signals": self.n_total_signals,
            "n_protected_wins": self.n_protected_wins,
            "n_winners": self.n_winners,
            "experiment_tag": self.experiment_tag,
        }

    @classmethod
    def from_trades(cls, trades: List[Trade], tag: str = "default") -> "TradeStats":
        """Build a TradeStats from a list of Trade objects."""
        if not trades:
            return cls(experiment_tag=tag)

        n_trades   = len(trades)
        pnl_list   = [t.pnl_frac for t in trades]
        n_wins     = sum(1 for p in pnl_list if p > 0)
        win_rate   = n_wins / n_trades if n_trades else 0.0
        avg_pnl    = float(np.mean(pnl_list)) if pnl_list else 0.0
        avg_hold   = float(np.mean([t.hold_bars for t in trades])) if trades else 0.0

        # Boost hours: typical high-activity windows (UTC)
        _BOOST_HOURS = set(range(7, 12)) | set(range(14, 18))
        boosted_hour_pnl = [t.pnl_frac for t in trades if t.entry_hour in _BOOST_HOURS]

        n_entered        = sum(t.was_entered for t in trades)
        n_total_signals  = n_trades  # each row represents one signal opportunity
        n_protected_wins = sum(t.was_protected for t in trades)
        n_winners        = sum(t.was_winner for t in trades)

        return cls(
            n_trades=n_trades,
            n_wins=n_wins,
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            avg_hold=avg_hold,
            pnl_list=pnl_list,
            boosted_hour_pnl=boosted_hour_pnl,
            n_entered=n_entered,
            n_total_signals=n_total_signals,
            n_protected_wins=n_protected_wins,
            n_winners=n_winners,
            experiment_tag=tag,
        )


# ---------------------------------------------------------------------------
# Database loader
# ---------------------------------------------------------------------------

def _col(df: pd.DataFrame, name: str, default):
    """Return column *name* from *df*, or a constant Series if missing."""
    if name in df.columns:
        return df[name]
    logger.debug("Column %r not found in trades table; using default %r", name, default)
    return pd.Series([default] * len(df), index=df.index)


def _load_df(db_path: Path, since_timestamp: Optional[str] = None) -> pd.DataFrame:
    """
    Load the trades table from SQLite into a DataFrame.

    Parameters
    ----------
    db_path        : path to the SQLite database file.
    since_timestamp: if provided, only load trades with timestamp >= this value.

    Returns
    -------
    DataFrame with one row per trade.  May be empty if the table is missing
    or the database does not exist yet.
    """
    if not db_path.exists():
        logger.warning("live_trades.db not found at %s; returning empty DataFrame.", db_path)
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(str(db_path))
        query = "SELECT * FROM trades"
        params: list = []
        if since_timestamp:
            query += " WHERE timestamp >= ?"
            params.append(since_timestamp)
        df = pd.read_sql_query(query, conn, params=params or None)
        conn.close()
        return df
    except Exception as exc:
        logger.error("Failed to load live_trades.db: %s", exc)
        return pd.DataFrame()


def _df_to_trades(df: pd.DataFrame) -> List[Trade]:
    """Convert a DataFrame to a list of Trade objects."""
    if df.empty:
        return []

    trades = []
    for _, row in df.iterrows():
        param_snap: dict = {}
        raw = _col(df, "param_snapshot", "{}").iloc[0] if "param_snapshot" in df.columns else "{}"
        try:
            param_snap = json.loads(row.get("param_snapshot", "{}") or "{}")
        except (json.JSONDecodeError, TypeError):
            param_snap = {}

        trades.append(Trade(
            id=int(row.get("id", 0)),
            timestamp=str(row.get("timestamp", "")),
            symbol=str(row.get("symbol", "UNKNOWN")),
            entry_price=float(row.get("entry_price", 0.0) or 0.0),
            exit_price=float(row.get("exit_price", 0.0) or 0.0),
            pnl_frac=float(row.get("pnl_frac", 0.0) or 0.0),
            hold_bars=int(row.get("hold_bars", 0) or 0),
            entry_hour=int(row.get("entry_hour", 0) or 0),
            signal_hour=int(row.get("signal_hour", 0) or 0),
            was_entered=int(row.get("was_entered", 1) or 1),
            was_protected=int(row.get("was_protected", 0) or 0),
            was_winner=int(row.get("was_winner", 0) or 0),
            experiment_tag=str(row.get("experiment_tag", "default") or "default"),
            param_snapshot=param_snap,
        ))
    return trades


# ---------------------------------------------------------------------------
# TradeBatcher
# ---------------------------------------------------------------------------

class TradeBatcher:
    """
    Load and batch live trades for Bayesian updating.

    Parameters
    ----------
    db_path : path to live_trades.db.  Defaults to the standard location
              relative to this file.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _DEFAULT_DB

    def load_all(self, since_timestamp: Optional[str] = None) -> List[Trade]:
        """
        Load all trades from the database.

        Parameters
        ----------
        since_timestamp : ISO-8601 timestamp string.  Only trades newer
                          than this are returned (for incremental updates).
        """
        df = _load_df(self.db_path, since_timestamp=since_timestamp)
        trades = _df_to_trades(df)
        logger.info("Loaded %d trades from %s", len(trades), self.db_path)
        return trades

    def batch_all(self, since_timestamp: Optional[str] = None) -> TradeStats:
        """
        Return a single TradeStats aggregated over all trades.

        Parameters
        ----------
        since_timestamp : only include trades after this timestamp.
        """
        trades = self.load_all(since_timestamp=since_timestamp)
        return TradeStats.from_trades(trades, tag="all")

    def batch_by_experiment(
        self, since_timestamp: Optional[str] = None
    ) -> Dict[str, TradeStats]:
        """
        Return a dict of TradeStats, one per experiment_tag.

        Useful for A/B experiments where each arm used different parameter
        settings.  The returned dict is keyed by experiment_tag.
        """
        trades = self.load_all(since_timestamp=since_timestamp)
        groups: Dict[str, List[Trade]] = {}
        for t in trades:
            groups.setdefault(t.experiment_tag, []).append(t)
        return {tag: TradeStats.from_trades(ts, tag=tag) for tag, ts in groups.items()}

    def get_recent_pnl(
        self,
        n: int = 100,
        since_timestamp: Optional[str] = None,
    ) -> np.ndarray:
        """
        Return an array of the *n* most recent trade P&L values.

        Parameters
        ----------
        n               : maximum number of recent trades to return.
        since_timestamp : filter applied before taking the last *n*.
        """
        trades = self.load_all(since_timestamp=since_timestamp)
        pnl = [t.pnl_frac for t in trades]
        return np.array(pnl[-n:], dtype=float)

    def summary(self) -> dict:
        """Return a quick summary dict without loading all trades."""
        stats = self.batch_all()
        return {
            "db_path": str(self.db_path),
            "n_trades": stats.n_trades,
            "win_rate": round(stats.win_rate, 4),
            "avg_pnl": round(stats.avg_pnl, 6),
            "avg_hold": round(stats.avg_hold, 1),
        }


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def load_live_trades(
    db_path: Optional[Path] = None,
    since_timestamp: Optional[str] = None,
) -> List[Trade]:
    """
    Load live trades from the standard database location.

    Parameters
    ----------
    db_path         : override default DB path.
    since_timestamp : ISO-8601 string; only return trades newer than this.

    Returns
    -------
    List of Trade objects.
    """
    batcher = TradeBatcher(db_path=db_path)
    return batcher.load_all(since_timestamp=since_timestamp)
