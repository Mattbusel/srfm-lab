"""
execution/risk/attribution.py
==============================
P&L attribution by signal factor for the SRFM Lab trading system.

Each completed trade's P&L is decomposed into contributions from the
strategy's named signal factors. The decomposition replays the signal
states that were active at trade entry and exit using data stored in
the nav_state table and signal_overrides.json.

Factor definitions (AttributionFactor enum) map directly to the
strategy components described in live_trader_alpaca.py.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("execution.risk.attribution")

_DB_PATH = Path(__file__).parents[2] / "execution" / "live_trades.db"
_OVERRIDES_FILE = Path(__file__).parents[2] / "config" / "signal_overrides.json"


# ---------------------------------------------------------------------------
# Signal factor taxonomy
# ---------------------------------------------------------------------------

class AttributionFactor(str, Enum):
    """Named signal factors contributing to trade P&L."""
    BH_SIGNAL     = "bh_signal"       # Black-Hole physics entry signal
    GARCH_SIZING  = "garch_sizing"     # GARCH vol-scaled position size effect
    OU_OVERLAY    = "ou_overlay"       # Ornstein-Uhlenbeck mean-reversion overlay
    GRANGER_BOOST = "granger_boost"    # Network Granger-causality signal boost
    ML_SIGNAL     = "ml_signal"        # ML signal module boost / suppress
    EVENT_CAL     = "event_cal"        # EventCalendarFilter timing effect
    QUAT_NAV      = "quat_nav"         # Quaternion navigation sizing modifier
    MARKET        = "market"           # Broad market beta (SPY proxy)
    RESIDUAL      = "residual"         # Unexplained residual


ALL_FACTORS = list(AttributionFactor)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TradeAttribution:
    """P&L attribution for a single completed trade."""
    trade_id: int
    symbol: str
    factors: Dict[AttributionFactor, float]   # factor -> USD contribution
    total_pnl: float
    entry_time: str
    exit_time: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def attributed_pnl(self) -> float:
        """Sum of all factor contributions."""
        return sum(self.factors.values())

    @property
    def residual(self) -> float:
        """Rounding residual between total_pnl and attributed sum."""
        return self.total_pnl - self.attributed_pnl


@dataclass
class FactorPerformance:
    """Aggregated performance statistics for a single factor."""
    factor: AttributionFactor
    cumulative_pnl: float
    sharpe: float               # annualised Sharpe (daily attribution series)
    max_dd: float               # maximum drawdown of factor P&L curve
    n_trades: int
    win_rate: float             # fraction of trades where factor contribution > 0


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def read_live_trades(db_path: Path = _DB_PATH) -> pd.DataFrame:
    """
    Read all rows from live_trades table and return as a DataFrame.

    Columns: id, symbol, side, qty, price, notional, fill_time,
             order_id, strategy_version
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            df = pd.read_sql_query("SELECT * FROM live_trades ORDER BY fill_time ASC", conn)
        return df
    except Exception as exc:
        log.error("read_live_trades failed: %s", exc)
        return pd.DataFrame()


def _read_trade_pnl(db_path: Path) -> pd.DataFrame:
    """Read completed trade P&L records."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            df = pd.read_sql_query("SELECT * FROM trade_pnl ORDER BY entry_time ASC", conn)
        return df
    except Exception as exc:
        log.error("_read_trade_pnl failed: %s", exc)
        return pd.DataFrame()


def _read_nav_state(db_path: Path, symbol: str) -> pd.DataFrame:
    """Read nav_state rows for a given symbol, sorted by bar_time."""
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM nav_state WHERE symbol=? ORDER BY bar_time ASC",
                conn,
                params=(symbol,),
            )
        return df
    except Exception as exc:
        log.error("_read_nav_state failed for %s: %s", symbol, exc)
        return pd.DataFrame()


def _load_overrides(overrides_file: Path) -> Dict:
    """Load signal_overrides.json; return empty dict if unavailable."""
    if not overrides_file.exists():
        return {}
    try:
        with open(overrides_file, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        log.warning("Could not read signal_overrides: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Factor weight reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_factor_weights(
    symbol: str,
    entry_time: str,
    exit_time: str,
    nav_df: pd.DataFrame,
    overrides: Dict,
) -> Dict[AttributionFactor, float]:
    """
    Reconstruct approximate factor weight fractions for a trade interval.

    The weights are used to apportion the trade's raw P&L across factors.
    They are derived from the nav_state signals closest in time to the
    entry and exit bars.

    Returns a dict mapping factor -> weight (fractions that sum to 1.0).
    All weights are non-negative; factors that were inactive get 0.
    """
    weights: Dict[AttributionFactor, float] = {f: 0.0 for f in AttributionFactor}

    # Locate the nav_state row closest to entry_time
    if nav_df.empty:
        weights[AttributionFactor.BH_SIGNAL] = 1.0
        weights[AttributionFactor.RESIDUAL] = 0.0
        return _normalise_weights(weights)

    try:
        entry_ts = pd.Timestamp(entry_time)
        nav_df["_bt"] = pd.to_datetime(nav_df["bar_time"], utc=True)
        delta = (nav_df["_bt"] - entry_ts).abs()
        closest_idx = delta.idxmin()
        row = nav_df.loc[closest_idx]
    except Exception:
        weights[AttributionFactor.BH_SIGNAL] = 1.0
        return _normalise_weights(weights)

    # BH signal contribution proportional to bh_mass
    bh_mass = float(row.get("bh_mass", 0.0))
    bh_active = int(row.get("bh_active", 0))
    bh_raw = bh_mass * (1.5 if bh_active else 0.5)
    weights[AttributionFactor.BH_SIGNAL] = max(0.0, bh_raw)

    # QuatNav: angular velocity and geodesic deviation modulate sizing
    ang_vel = float(row.get("angular_velocity", 0.0))
    geo_dev = float(row.get("geodesic_deviation", 0.0))
    quat_raw = max(0.0, 1.0 - min(ang_vel / (geo_dev + 1e-6), 1.0))
    weights[AttributionFactor.QUAT_NAV] = quat_raw * 0.5  # up to 50% contribution

    # Lorentz / GARCH sizing modifier
    lorentz_flag = int(row.get("lorentz_boost_applied", 0))
    rapidity = float(row.get("lorentz_boost_rapidity", 0.0))
    garch_raw = abs(math.tanh(rapidity)) if lorentz_flag else 0.1
    weights[AttributionFactor.GARCH_SIZING] = max(0.0, garch_raw)

    # OU overlay: constant baseline per active trade
    weights[AttributionFactor.OU_OVERLAY] = 0.15

    # Overrides can signal Granger boost, ML signal, event calendar
    sym_override = overrides.get(symbol, {})
    if sym_override.get("granger_boost", False):
        weights[AttributionFactor.GRANGER_BOOST] = 0.20
    if sym_override.get("ml_signal", 0.0) != 0.0:
        ml_v = float(sym_override["ml_signal"])
        weights[AttributionFactor.ML_SIGNAL] = abs(ml_v) * 0.30
    if sym_override.get("event_calendar_active", False):
        weights[AttributionFactor.EVENT_CAL] = 0.10

    # Market beta proxy: always allocates a baseline share
    weights[AttributionFactor.MARKET] = 0.20

    # Residual catches remainder
    weights[AttributionFactor.RESIDUAL] = 0.05

    return _normalise_weights(weights)


def _normalise_weights(w: Dict[AttributionFactor, float]) -> Dict[AttributionFactor, float]:
    """Ensure factor weights are non-negative and sum to 1.0."""
    total = sum(v for v in w.values() if v > 0)
    if total <= 0:
        w[AttributionFactor.RESIDUAL] = 1.0
        total = 1.0
    return {k: max(0.0, v) / total for k, v in w.items()}


# ---------------------------------------------------------------------------
# PnLAttributor
# ---------------------------------------------------------------------------

_CREATE_ATTRIBUTION_SQL = """
CREATE TABLE IF NOT EXISTS attribution (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id        INTEGER,
    symbol          TEXT,
    entry_time      TEXT,
    exit_time       TEXT,
    total_pnl       REAL,
    bh_signal       REAL,
    garch_sizing    REAL,
    ou_overlay      REAL,
    granger_boost   REAL,
    ml_signal       REAL,
    event_cal       REAL,
    quat_nav        REAL,
    market          REAL,
    residual        REAL,
    computed_at     TEXT
);
"""


class PnLAttributor:
    """
    Reads completed trades from trade_pnl, reconstructs signal states
    from nav_state at entry/exit times, and decomposes P&L into factor
    contributions.

    Attribution logic:
        1. Load nav_state rows closest to entry and exit bars.
        2. Reconstruct factor weight fractions from signal intensities.
        3. Scale weights by total_pnl to produce dollar attributions.
        4. Persist to attribution table.
    """

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self.db_path = db_path
        self._overrides: Dict = _load_overrides(_OVERRIDES_FILE)
        self._overrides_mtime: float = 0.0
        self._ensure_table()

    def _ensure_table(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_ATTRIBUTION_SQL)
            conn.commit()

    def _refresh_overrides(self) -> None:
        try:
            mtime = _OVERRIDES_FILE.stat().st_mtime
            if mtime != self._overrides_mtime:
                self._overrides = _load_overrides(_OVERRIDES_FILE)
                self._overrides_mtime = mtime
        except Exception:
            pass

    def _already_attributed(self, trade_id: int) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT id FROM attribution WHERE trade_id=?", (trade_id,)
                ).fetchone()
            return row is not None
        except Exception:
            return False

    def attribute_trade(self, trade_row: pd.Series) -> TradeAttribution:
        """
        Compute factor attribution for a single trade_pnl row.

        Parameters
        ----------
        trade_row : one row from the trade_pnl DataFrame
        """
        trade_id = int(trade_row.get("id", 0))
        symbol = str(trade_row["symbol"])
        entry_time = str(trade_row.get("entry_time", ""))
        exit_time = str(trade_row.get("exit_time", ""))
        total_pnl = float(trade_row.get("pnl", 0.0))

        nav_df = _read_nav_state(self.db_path, symbol)
        weights = _reconstruct_factor_weights(
            symbol, entry_time, exit_time, nav_df, self._overrides
        )

        # Dollar contribution = weight * total_pnl
        factor_pnl = {f: weights.get(f, 0.0) * total_pnl for f in AttributionFactor}

        return TradeAttribution(
            trade_id=trade_id,
            symbol=symbol,
            factors=factor_pnl,
            total_pnl=total_pnl,
            entry_time=entry_time,
            exit_time=exit_time,
        )

    def run(self, force_recompute: bool = False) -> List[TradeAttribution]:
        """
        Attribute all unprocessed trades in trade_pnl.

        Returns list of TradeAttribution for newly processed trades.
        """
        self._refresh_overrides()
        trade_df = _read_trade_pnl(self.db_path)
        if trade_df.empty:
            log.info("No trade_pnl rows to attribute.")
            return []

        results: List[TradeAttribution] = []
        for _, row in trade_df.iterrows():
            trade_id = int(row.get("id", 0))
            if not force_recompute and self._already_attributed(trade_id):
                continue
            ta = self.attribute_trade(row)
            self._persist(ta)
            results.append(ta)

        log.info("Attributed %d new trades.", len(results))
        return results

    def _persist(self, ta: TradeAttribution) -> None:
        computed_at = datetime.now(timezone.utc).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO attribution
                       (trade_id, symbol, entry_time, exit_time, total_pnl,
                        bh_signal, garch_sizing, ou_overlay, granger_boost,
                        ml_signal, event_cal, quat_nav, market, residual,
                        computed_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        ta.trade_id, ta.symbol, ta.entry_time, ta.exit_time,
                        ta.total_pnl,
                        ta.factors.get(AttributionFactor.BH_SIGNAL, 0.0),
                        ta.factors.get(AttributionFactor.GARCH_SIZING, 0.0),
                        ta.factors.get(AttributionFactor.OU_OVERLAY, 0.0),
                        ta.factors.get(AttributionFactor.GRANGER_BOOST, 0.0),
                        ta.factors.get(AttributionFactor.ML_SIGNAL, 0.0),
                        ta.factors.get(AttributionFactor.EVENT_CAL, 0.0),
                        ta.factors.get(AttributionFactor.QUAT_NAV, 0.0),
                        ta.factors.get(AttributionFactor.MARKET, 0.0),
                        ta.factors.get(AttributionFactor.RESIDUAL, 0.0),
                        computed_at,
                    ),
                )
                conn.commit()
        except sqlite3.Error as exc:
            log.error("Failed to persist attribution for trade %d: %s", ta.trade_id, exc)


# ---------------------------------------------------------------------------
# Attribution Report
# ---------------------------------------------------------------------------

def _max_drawdown(pnl_series: np.ndarray) -> float:
    """Maximum drawdown of a cumulative P&L curve."""
    if len(pnl_series) == 0:
        return 0.0
    cum = np.cumsum(pnl_series)
    running_max = np.maximum.accumulate(cum)
    drawdowns = running_max - cum
    return float(drawdowns.max())


def _sharpe(pnl_series: np.ndarray, periods_per_year: float = 252.0) -> float:
    """Annualised Sharpe ratio from a series of per-trade P&L values."""
    if len(pnl_series) < 2:
        return 0.0
    mean = np.mean(pnl_series)
    std = np.std(pnl_series, ddof=1)
    if std <= 0:
        return 0.0
    return float(mean / std * math.sqrt(periods_per_year))


class AttributionReport:
    """
    Aggregates TradeAttribution records into FactorPerformance statistics.

    Can filter by a rolling date window (days parameter).
    """

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self.db_path = db_path

    def _load_attributions(self, days: Optional[int] = None) -> pd.DataFrame:
        """Load attribution table, optionally filtered to last N days."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                df = pd.read_sql_query(
                    "SELECT * FROM attribution ORDER BY entry_time ASC", conn
                )
        except Exception as exc:
            log.error("_load_attributions failed: %s", exc)
            return pd.DataFrame()

        if df.empty:
            return df

        if days is not None:
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
            try:
                df["entry_ts"] = pd.to_datetime(df["entry_time"], utc=True)
                df = df[df["entry_ts"] >= cutoff]
            except Exception:
                pass

        return df

    def factor_performance(self, days: Optional[int] = None) -> List[FactorPerformance]:
        """
        Compute FactorPerformance for each AttributionFactor.

        Parameters
        ----------
        days : if provided, restrict to last N calendar days
        """
        df = self._load_attributions(days=days)
        if df.empty:
            return []

        results: List[FactorPerformance] = []
        factor_cols = {
            AttributionFactor.BH_SIGNAL:     "bh_signal",
            AttributionFactor.GARCH_SIZING:  "garch_sizing",
            AttributionFactor.OU_OVERLAY:    "ou_overlay",
            AttributionFactor.GRANGER_BOOST: "granger_boost",
            AttributionFactor.ML_SIGNAL:     "ml_signal",
            AttributionFactor.EVENT_CAL:     "event_cal",
            AttributionFactor.QUAT_NAV:      "quat_nav",
            AttributionFactor.MARKET:        "market",
            AttributionFactor.RESIDUAL:      "residual",
        }

        for factor, col in factor_cols.items():
            if col not in df.columns:
                continue
            pnl_series = df[col].fillna(0.0).values.astype(float)
            n_trades = len(pnl_series)
            win_rate = float((pnl_series > 0).mean()) if n_trades > 0 else 0.0

            results.append(FactorPerformance(
                factor=factor,
                cumulative_pnl=float(pnl_series.sum()),
                sharpe=_sharpe(pnl_series),
                max_dd=_max_drawdown(pnl_series),
                n_trades=n_trades,
                win_rate=win_rate,
            ))

        return results

    def best_worst_factors(
        self, days: Optional[int] = None
    ) -> Tuple[Optional[FactorPerformance], Optional[FactorPerformance]]:
        """
        Return (best_factor, worst_factor) by cumulative P&L.
        """
        perf = self.factor_performance(days=days)
        if not perf:
            return None, None
        best = max(perf, key=lambda fp: fp.cumulative_pnl)
        worst = min(perf, key=lambda fp: fp.cumulative_pnl)
        return best, worst

    def rolling_factor_pnl(
        self, window_days: int = 30
    ) -> pd.DataFrame:
        """
        Compute rolling N-day cumulative P&L per factor.

        Returns DataFrame with columns: entry_time, factor, rolling_pnl
        """
        df = self._load_attributions()
        if df.empty:
            return pd.DataFrame()

        factor_cols = [
            "bh_signal", "garch_sizing", "ou_overlay", "granger_boost",
            "ml_signal", "event_cal", "quat_nav", "market", "residual",
        ]
        available = [c for c in factor_cols if c in df.columns]
        if not available:
            return pd.DataFrame()

        try:
            df["entry_ts"] = pd.to_datetime(df["entry_time"], utc=True)
            df = df.sort_values("entry_ts")
            rolling_rows = []
            for col in available:
                series = df.set_index("entry_ts")[col].fillna(0.0)
                rolling = series.rolling(f"{window_days}D").sum()
                for ts, val in rolling.items():
                    rolling_rows.append({"entry_time": ts, "factor": col, "rolling_pnl": val})
            return pd.DataFrame(rolling_rows)
        except Exception as exc:
            log.error("rolling_factor_pnl failed: %s", exc)
            return pd.DataFrame()

    def summary_dict(self, days: Optional[int] = None) -> Dict:
        """Return a JSON-serialisable summary of factor performance."""
        perf = self.factor_performance(days=days)
        best, worst = self.best_worst_factors(days=days)
        return {
            "factors": [
                {
                    "factor": fp.factor.value,
                    "cumulative_pnl": round(fp.cumulative_pnl, 2),
                    "sharpe": round(fp.sharpe, 4),
                    "max_dd": round(fp.max_dd, 2),
                    "n_trades": fp.n_trades,
                    "win_rate": round(fp.win_rate, 4),
                }
                for fp in perf
            ],
            "best_factor": best.factor.value if best else None,
            "worst_factor": worst.factor.value if worst else None,
        }
