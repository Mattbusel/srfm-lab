"""
post_trade_analyzer.py -- Comprehensive post-trade analysis for LARSA v18.

Analyzes completed trades to identify entry quality, exit timing, MFE/MAE
patterns, and missed opportunities. Provides ExitQualityAnalyzer for comparing
RL exits vs simpler alternatives.
"""

from __future__ import annotations

import sqlite3
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PostTradeRecord:
    """Single completed trade with all enrichment fields."""
    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    hold_bars: int
    entry_reason: str          # bh_mass / cf_cross / hurst_signal / manual
    exit_reason: str           # rl_exit / time / signal / stop
    bh_mass_at_entry: float
    hurst_at_entry: float
    nav_omega_at_entry: float
    garch_vol_at_entry: float
    mfe: float = 0.0           # Maximum Favorable Excursion (filled after load)
    mae: float = 0.0           # Maximum Adverse Excursion (filled after load)
    max_pnl_during: float = 0.0  # peak unrealized P&L
    min_pnl_during: float = 0.0  # trough unrealized P&L
    side: str = "long"         # long / short
    nav_filtered: bool = False  # True if NAV gate was active
    event_filtered: bool = False  # True if event calendar blocked entry

    @property
    def win(self) -> bool:
        return self.pnl > 0.0

    @property
    def exit_efficiency(self) -> float:
        """Fraction of MFE captured. 1.0 = exited at peak."""
        if self.mfe <= 0.0:
            return 0.0
        return min(1.0, max(-1.0, self.pnl_pct / self.mfe))

    @property
    def mfe_mae_ratio(self) -> float:
        """Higher ratio = trade had better upside vs downside."""
        if abs(self.mae) < 1e-9:
            return float("inf") if self.mfe > 0 else 0.0
        return abs(self.mfe / self.mae)


# ---------------------------------------------------------------------------
# Database loader helpers
# ---------------------------------------------------------------------------

_CREATE_TRADES_DDL = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id        TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    entry_time      TEXT NOT NULL,
    exit_time       TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    exit_price      REAL NOT NULL,
    qty             REAL NOT NULL,
    pnl             REAL NOT NULL,
    pnl_pct         REAL NOT NULL,
    hold_bars       INTEGER NOT NULL,
    entry_reason    TEXT NOT NULL,
    exit_reason     TEXT NOT NULL,
    bh_mass_at_entry     REAL NOT NULL DEFAULT 0.0,
    hurst_at_entry       REAL NOT NULL DEFAULT 0.5,
    nav_omega_at_entry   REAL NOT NULL DEFAULT 1.0,
    garch_vol_at_entry   REAL NOT NULL DEFAULT 0.02,
    mfe                  REAL NOT NULL DEFAULT 0.0,
    mae                  REAL NOT NULL DEFAULT 0.0,
    max_pnl_during       REAL NOT NULL DEFAULT 0.0,
    min_pnl_during       REAL NOT NULL DEFAULT 0.0,
    side                 TEXT NOT NULL DEFAULT 'long',
    nav_filtered         INTEGER NOT NULL DEFAULT 0,
    event_filtered       INTEGER NOT NULL DEFAULT 0
);
"""


def _row_to_record(row: dict) -> PostTradeRecord:
    def _dt(s: str) -> datetime:
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return datetime.fromisoformat(s)

    return PostTradeRecord(
        trade_id=row["trade_id"],
        symbol=row["symbol"],
        entry_time=_dt(row["entry_time"]),
        exit_time=_dt(row["exit_time"]),
        entry_price=float(row["entry_price"]),
        exit_price=float(row["exit_price"]),
        qty=float(row["qty"]),
        pnl=float(row["pnl"]),
        pnl_pct=float(row["pnl_pct"]),
        hold_bars=int(row["hold_bars"]),
        entry_reason=row["entry_reason"],
        exit_reason=row["exit_reason"],
        bh_mass_at_entry=float(row.get("bh_mass_at_entry", 0.0)),
        hurst_at_entry=float(row.get("hurst_at_entry", 0.5)),
        nav_omega_at_entry=float(row.get("nav_omega_at_entry", 1.0)),
        garch_vol_at_entry=float(row.get("garch_vol_at_entry", 0.02)),
        mfe=float(row.get("mfe", 0.0)),
        mae=float(row.get("mae", 0.0)),
        max_pnl_during=float(row.get("max_pnl_during", 0.0)),
        min_pnl_during=float(row.get("min_pnl_during", 0.0)),
        side=row.get("side", "long"),
        nav_filtered=bool(row.get("nav_filtered", 0)),
        event_filtered=bool(row.get("event_filtered", 0)),
    )


def load_trades_from_db(db_path: str | Path) -> list[PostTradeRecord]:
    """Load all trades from SQLite. Creates table if missing."""
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(_CREATE_TRADES_DDL)
        conn.commit()
        rows = conn.execute("SELECT * FROM trades ORDER BY entry_time").fetchall()
        records = [_row_to_record(dict(r)) for r in rows]
        logger.info("Loaded %d trades from %s", len(records), db_path)
        return records
    finally:
        conn.close()


def save_trades_to_db(trades: list[PostTradeRecord], db_path: str | Path) -> None:
    """Upsert PostTradeRecords into SQLite."""
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(_CREATE_TRADES_DDL)
        for t in trades:
            conn.execute(
                """
                INSERT OR REPLACE INTO trades VALUES (
                    :trade_id, :symbol, :entry_time, :exit_time,
                    :entry_price, :exit_price, :qty,
                    :pnl, :pnl_pct, :hold_bars,
                    :entry_reason, :exit_reason,
                    :bh_mass_at_entry, :hurst_at_entry,
                    :nav_omega_at_entry, :garch_vol_at_entry,
                    :mfe, :mae, :max_pnl_during, :min_pnl_during,
                    :side, :nav_filtered, :event_filtered
                )
                """,
                {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "qty": t.qty,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "hold_bars": t.hold_bars,
                    "entry_reason": t.entry_reason,
                    "exit_reason": t.exit_reason,
                    "bh_mass_at_entry": t.bh_mass_at_entry,
                    "hurst_at_entry": t.hurst_at_entry,
                    "nav_omega_at_entry": t.nav_omega_at_entry,
                    "garch_vol_at_entry": t.garch_vol_at_entry,
                    "mfe": t.mfe,
                    "mae": t.mae,
                    "max_pnl_during": t.max_pnl_during,
                    "min_pnl_during": t.min_pnl_during,
                    "side": t.side,
                    "nav_filtered": int(t.nav_filtered),
                    "event_filtered": int(t.event_filtered),
                },
            )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Utility statistics helpers
# ---------------------------------------------------------------------------

def _sharpe(pnls: list[float], periods_per_year: int = 252) -> float:
    arr = np.array(pnls, dtype=float)
    if len(arr) < 2:
        return 0.0
    std = np.std(arr, ddof=1)
    if std < 1e-12:
        return 0.0
    return float(np.mean(arr) / std * np.sqrt(periods_per_year))


def _win_rate(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    return float(sum(1 for p in pnls if p > 0) / len(pnls))


def _avg(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _bucket_label(value: float, thresholds: list[float], labels: list[str]) -> str:
    for threshold, label in zip(thresholds, labels):
        if value < threshold:
            return label
    return labels[-1]


# ---------------------------------------------------------------------------
# PostTradeAnalyzer
# ---------------------------------------------------------------------------

class PostTradeAnalyzer:
    """
    Primary post-trade analysis engine for LARSA v18.

    Usage::

        analyzer = PostTradeAnalyzer("trades.db")
        eq = analyzer.analyze_entry_quality()
        mfe = analyzer.analyze_mfe_mae()
    """

    # BH mass quintile thresholds (calibrated to typical LARSA output range 0-1)
    BH_MASS_THRESHOLDS = [0.2, 0.4, 0.6, 0.8, 1.0]
    BH_MASS_LABELS = ["q1_low", "q2", "q3", "q4", "q5_high"]

    HURST_THRESHOLDS = [0.4, 0.55, 1.0]
    HURST_LABELS = ["mean_reverting", "neutral", "trending"]

    GARCH_VOL_THRESHOLDS = [0.015, 0.03, 1.0]
    GARCH_VOL_LABELS = ["low_vol", "med_vol", "high_vol"]

    def __init__(self, db_path: str | Path | None = None,
                 trades: list[PostTradeRecord] | None = None):
        if trades is not None:
            self.trades = trades
        elif db_path is not None:
            self.trades = load_trades_from_db(db_path)
        else:
            self.trades = []
        logger.info("PostTradeAnalyzer initialized with %d trades", len(self.trades))

    # ------------------------------------------------------------------
    # Entry quality
    # ------------------------------------------------------------------

    def analyze_entry_quality(self) -> dict:
        """
        For each entry condition (BH mass level, Hurst regime, entry reason),
        compute win rate and average P&L.

        Returns dict with keys: bh_mass_level, hurst_regime, entry_reason,
        nav_omega_level, garch_vol_regime.
        """
        result = {}

        # -- BH mass quintiles
        bh_groups: dict[str, list[float]] = {lb: [] for lb in self.BH_MASS_LABELS}
        for t in self.trades:
            lb = _bucket_label(t.bh_mass_at_entry, self.BH_MASS_THRESHOLDS,
                               self.BH_MASS_LABELS)
            bh_groups[lb].append(t.pnl_pct)

        result["bh_mass_level"] = {
            lb: {
                "n": len(pnls),
                "win_rate": _win_rate(pnls),
                "avg_pnl_pct": _avg(pnls),
                "sharpe": _sharpe(pnls),
            }
            for lb, pnls in bh_groups.items()
        }

        # -- Hurst regime
        hurst_groups: dict[str, list[float]] = {lb: [] for lb in self.HURST_LABELS}
        for t in self.trades:
            lb = _bucket_label(t.hurst_at_entry, self.HURST_THRESHOLDS,
                               self.HURST_LABELS)
            hurst_groups[lb].append(t.pnl_pct)

        result["hurst_regime"] = {
            lb: {
                "n": len(pnls),
                "win_rate": _win_rate(pnls),
                "avg_pnl_pct": _avg(pnls),
                "sharpe": _sharpe(pnls),
            }
            for lb, pnls in hurst_groups.items()
        }

        # -- Entry reason
        reason_groups: dict[str, list[float]] = {}
        for t in self.trades:
            reason_groups.setdefault(t.entry_reason, []).append(t.pnl_pct)

        result["entry_reason"] = {
            reason: {
                "n": len(pnls),
                "win_rate": _win_rate(pnls),
                "avg_pnl_pct": _avg(pnls),
                "sharpe": _sharpe(pnls),
            }
            for reason, pnls in reason_groups.items()
        }

        # -- NAV omega level (quartiles 0-1)
        nav_thresholds = [0.25, 0.5, 0.75, 1.0]
        nav_labels = ["q1_low", "q2", "q3", "q4_high"]
        nav_groups: dict[str, list[float]] = {lb: [] for lb in nav_labels}
        for t in self.trades:
            lb = _bucket_label(t.nav_omega_at_entry, nav_thresholds, nav_labels)
            nav_groups[lb].append(t.pnl_pct)

        result["nav_omega_level"] = {
            lb: {
                "n": len(pnls),
                "win_rate": _win_rate(pnls),
                "avg_pnl_pct": _avg(pnls),
                "sharpe": _sharpe(pnls),
            }
            for lb, pnls in nav_groups.items()
        }

        # -- GARCH vol regime
        garch_groups: dict[str, list[float]] = {lb: [] for lb in self.GARCH_VOL_LABELS}
        for t in self.trades:
            lb = _bucket_label(t.garch_vol_at_entry, self.GARCH_VOL_THRESHOLDS,
                               self.GARCH_VOL_LABELS)
            garch_groups[lb].append(t.pnl_pct)

        result["garch_vol_regime"] = {
            lb: {
                "n": len(pnls),
                "win_rate": _win_rate(pnls),
                "avg_pnl_pct": _avg(pnls),
                "sharpe": _sharpe(pnls),
            }
            for lb, pnls in garch_groups.items()
        }

        return result

    # ------------------------------------------------------------------
    # Exit timing
    # ------------------------------------------------------------------

    def analyze_exit_timing(self) -> dict:
        """
        For each exit reason, compute avg bars held, avg P&L, and
        P&L captured as fraction of MFE (exit efficiency).
        """
        groups: dict[str, list[PostTradeRecord]] = {}
        for t in self.trades:
            groups.setdefault(t.exit_reason, []).append(t)

        result = {}
        for reason, records in groups.items():
            bars = [r.hold_bars for r in records]
            pnls = [r.pnl_pct for r in records]
            efficiencies = [r.exit_efficiency for r in records if r.mfe > 0]
            result[reason] = {
                "n": len(records),
                "avg_hold_bars": _avg(bars),
                "median_hold_bars": float(np.median(bars)) if bars else 0.0,
                "avg_pnl_pct": _avg(pnls),
                "win_rate": _win_rate(pnls),
                "avg_exit_efficiency": _avg(efficiencies) if efficiencies else None,
                "pct_captured_vs_mfe": _avg(efficiencies) * 100.0 if efficiencies else None,
            }

        return result

    # ------------------------------------------------------------------
    # MFE / MAE analysis
    # ------------------------------------------------------------------

    def analyze_mfe_mae(self) -> dict:
        """
        Maximum Favorable Excursion and Maximum Adverse Excursion analysis.

        Returns stats on:
        - MFE distribution (how much upside was available)
        - MAE distribution (how much drawdown was experienced)
        - MFE/MAE ratio distribution
        - Exit efficiency distribution
        - Trades that gave back >50% of MFE (poor exits)
        """
        if not self.trades:
            return {}

        mfe_vals = [t.mfe for t in self.trades]
        mae_vals = [abs(t.mae) for t in self.trades]
        ratios = [t.mfe_mae_ratio for t in self.trades if not np.isinf(t.mfe_mae_ratio)]
        efficiencies = [t.exit_efficiency for t in self.trades if t.mfe > 0]

        # Trades that gave back more than 50% of their MFE
        gave_back = [
            t for t in self.trades
            if t.mfe > 0 and t.pnl < t.mfe * 0.5 and t.pnl >= 0
        ]
        # Trades that reversed from profitable to loss
        reversals = [
            t for t in self.trades
            if t.mfe > 0 and t.pnl < 0
        ]

        return {
            "mfe": {
                "mean": _avg(mfe_vals),
                "median": float(np.median(mfe_vals)),
                "p75": float(np.percentile(mfe_vals, 75)),
                "p95": float(np.percentile(mfe_vals, 95)),
            },
            "mae": {
                "mean": _avg(mae_vals),
                "median": float(np.median(mae_vals)),
                "p75": float(np.percentile(mae_vals, 75)),
                "p95": float(np.percentile(mae_vals, 95)),
            },
            "mfe_mae_ratio": {
                "mean": _avg(ratios),
                "median": float(np.median(ratios)) if ratios else 0.0,
                "interpretation": (
                    "good" if _avg(ratios) > 1.5
                    else "fair" if _avg(ratios) > 0.8
                    else "poor"
                ),
            },
            "exit_efficiency": {
                "mean": _avg(efficiencies),
                "median": float(np.median(efficiencies)) if efficiencies else 0.0,
                "pct_above_80pct": float(
                    sum(1 for e in efficiencies if e >= 0.8) / len(efficiencies)
                    if efficiencies else 0.0
                ),
            },
            "poor_exits": {
                "gave_back_gt50pct_mfe_n": len(gave_back),
                "gave_back_gt50pct_mfe_pct": len(gave_back) / len(self.trades) * 100.0,
                "full_reversals_n": len(reversals),
                "full_reversals_pct": len(reversals) / len(self.trades) * 100.0,
            },
            "total_trades": len(self.trades),
        }

    # ------------------------------------------------------------------
    # Missed opportunities
    # ------------------------------------------------------------------

    def find_missed_opportunities(
        self,
        bh_mass_threshold: float = 0.6,
    ) -> list[dict]:
        """
        Trades that were blocked by NAV gate or event calendar but had
        high BH mass -- potential missed opportunities.

        Parameters
        ----------
        bh_mass_threshold:
            Minimum BH mass to consider a missed opportunity.

        Returns list of dicts with trade_id, symbol, entry_time, block_reason,
        bh_mass_at_entry, estimated_pnl_pct (from cohort avg).
        """
        # Build average P&L by BH mass bucket for unfiltered trades
        unfiltered = [t for t in self.trades if not t.nav_filtered and not t.event_filtered]
        bucket_pnls: dict[str, list[float]] = {lb: [] for lb in self.BH_MASS_LABELS}
        for t in unfiltered:
            lb = _bucket_label(t.bh_mass_at_entry, self.BH_MASS_THRESHOLDS,
                               self.BH_MASS_LABELS)
            bucket_pnls[lb].append(t.pnl_pct)
        bucket_avg = {lb: _avg(pnls) for lb, pnls in bucket_pnls.items()}

        missed = []
        for t in self.trades:
            if t.bh_mass_at_entry < bh_mass_threshold:
                continue
            if not (t.nav_filtered or t.event_filtered):
                continue
            lb = _bucket_label(t.bh_mass_at_entry, self.BH_MASS_THRESHOLDS,
                               self.BH_MASS_LABELS)
            block = []
            if t.nav_filtered:
                block.append("nav_gate")
            if t.event_filtered:
                block.append("event_calendar")
            missed.append({
                "trade_id": t.trade_id,
                "symbol": t.symbol,
                "entry_time": t.entry_time.isoformat(),
                "block_reason": "+".join(block),
                "bh_mass_at_entry": t.bh_mass_at_entry,
                "hurst_at_entry": t.hurst_at_entry,
                "estimated_pnl_pct": bucket_avg.get(lb, 0.0),
            })

        missed.sort(key=lambda x: x["bh_mass_at_entry"], reverse=True)
        logger.info("Found %d missed opportunities (bh_mass >= %.2f)",
                    len(missed), bh_mass_threshold)
        return missed

    # ------------------------------------------------------------------
    # Cohort analysis
    # ------------------------------------------------------------------

    def cohort_analysis(self) -> pd.DataFrame:
        """
        Group trades by entry week. Return DataFrame with cohort Sharpe,
        win rate, avg P&L, and trade count.
        """
        if not self.trades:
            return pd.DataFrame()

        rows = []
        for t in self.trades:
            week_start = t.entry_time - timedelta(days=t.entry_time.weekday())
            week_str = week_start.strftime("%Y-W%V")
            rows.append({
                "cohort_week": week_str,
                "pnl_pct": t.pnl_pct,
                "win": int(t.win),
                "hold_bars": t.hold_bars,
                "exit_efficiency": t.exit_efficiency,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        grouped = df.groupby("cohort_week")
        cohort_df = grouped["pnl_pct"].agg(
            n="count",
            avg_pnl_pct="mean",
            std_pnl_pct="std",
        ).copy()
        cohort_df["win_rate"] = grouped["win"].mean()
        cohort_df["avg_hold_bars"] = grouped["hold_bars"].mean()
        cohort_df["avg_exit_efficiency"] = grouped["exit_efficiency"].mean()
        cohort_df["sharpe"] = (
            cohort_df["avg_pnl_pct"]
            / cohort_df["std_pnl_pct"].replace(0, np.nan)
            * np.sqrt(252)
        ).fillna(0.0)
        cohort_df = cohort_df.reset_index()
        cohort_df = cohort_df.sort_values("cohort_week").reset_index(drop=True)
        return cohort_df

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def summary_report(self) -> dict:
        """High-level summary covering all analysis dimensions."""
        if not self.trades:
            return {"error": "no trades loaded"}

        pnls = [t.pnl_pct for t in self.trades]
        return {
            "total_trades": len(self.trades),
            "win_rate": _win_rate(pnls),
            "avg_pnl_pct": _avg(pnls),
            "total_pnl_pct": float(sum(pnls)),
            "sharpe": _sharpe(pnls),
            "entry_quality": self.analyze_entry_quality(),
            "exit_timing": self.analyze_exit_timing(),
            "mfe_mae": self.analyze_mfe_mae(),
            "missed_opportunities_n": len(self.find_missed_opportunities()),
        }


# ---------------------------------------------------------------------------
# ExitQualityAnalyzer
# ---------------------------------------------------------------------------

class ExitQualityAnalyzer:
    """
    Compares actual exit outcomes (RL or signal-based) against hypothetical
    alternatives to diagnose whether the exit strategy is optimal.
    """

    def __init__(self, trades: list[PostTradeRecord]):
        self.trades = trades

    # ------------------------------------------------------------------
    # Exit efficiency per trade
    # ------------------------------------------------------------------

    def compute_exit_efficiencies(self) -> pd.DataFrame:
        """
        Return DataFrame with one row per trade including exit efficiency
        and supporting metrics.
        """
        rows = []
        for t in self.trades:
            rows.append({
                "trade_id": t.trade_id,
                "symbol": t.symbol,
                "exit_reason": t.exit_reason,
                "hold_bars": t.hold_bars,
                "pnl_pct": t.pnl_pct,
                "mfe": t.mfe,
                "mae": t.mae,
                "exit_efficiency": t.exit_efficiency,
                "mfe_mae_ratio": (
                    abs(t.mfe / t.mae)
                    if abs(t.mae) > 1e-9
                    else (float("inf") if t.mfe > 0 else 0.0)
                ),
                "gave_back_pct": (
                    (t.mfe - t.pnl) / t.mfe * 100.0
                    if t.mfe > 1e-9
                    else 0.0
                ),
                "bh_mass": t.bh_mass_at_entry,
                "hurst": t.hurst_at_entry,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # RL vs simple exits
    # ------------------------------------------------------------------

    def compare_rl_vs_simple_exits(
        self,
        simple_hold_bars: int = 5,
    ) -> dict:
        """
        Compare RL-based exits against a simple fixed-hold-time alternative.

        The "simple" strategy always exits at ``simple_hold_bars`` bars,
        approximating the P&L as pnl * (simple_hold_bars / hold_bars)
        when actual hold > simple_hold_bars, and actual pnl otherwise.

        Parameters
        ----------
        simple_hold_bars:
            Fixed number of bars the simple strategy holds before exiting.

        Returns dict with per-strategy stats.
        """
        rl_trades = [t for t in self.trades if t.exit_reason == "rl_exit"]
        other_trades = [t for t in self.trades if t.exit_reason != "rl_exit"]

        # Simple exit approximation: linear P&L ramp up to MFE then decay
        def simple_pnl_approx(t: PostTradeRecord, hold: int) -> float:
            if t.hold_bars == 0:
                return t.pnl_pct
            if hold >= t.hold_bars:
                return t.pnl_pct
            # Linearly scale P&L -- crude but reasonable for short holds
            return t.pnl_pct * (hold / t.hold_bars)

        all_rl_pnls = [t.pnl_pct for t in rl_trades]
        all_simple_pnls = [
            simple_pnl_approx(t, simple_hold_bars) for t in rl_trades
        ]

        return {
            "rl_exit": {
                "n": len(rl_trades),
                "avg_pnl_pct": _avg(all_rl_pnls),
                "win_rate": _win_rate(all_rl_pnls),
                "sharpe": _sharpe(all_rl_pnls),
                "avg_hold_bars": _avg([t.hold_bars for t in rl_trades]),
                "avg_exit_efficiency": _avg(
                    [t.exit_efficiency for t in rl_trades if t.mfe > 0]
                ),
            },
            f"simple_hold_{simple_hold_bars}bars": {
                "n": len(rl_trades),
                "avg_pnl_pct": _avg(all_simple_pnls),
                "win_rate": _win_rate(all_simple_pnls),
                "sharpe": _sharpe(all_simple_pnls),
                "avg_hold_bars": simple_hold_bars,
                "avg_exit_efficiency": None,
            },
            "non_rl_exits": {
                "n": len(other_trades),
                "avg_pnl_pct": _avg([t.pnl_pct for t in other_trades]),
                "win_rate": _win_rate([t.pnl_pct for t in other_trades]),
                "sharpe": _sharpe([t.pnl_pct for t in other_trades]),
                "avg_hold_bars": _avg([t.hold_bars for t in other_trades]),
            },
            "rl_advantage_pnl_pct": (
                _avg(all_rl_pnls) - _avg(all_simple_pnls)
            ),
        }

    # ------------------------------------------------------------------
    # Optimal hold bars by BH regime
    # ------------------------------------------------------------------

    def optimal_exit_bars(self) -> dict:
        """
        For each BH mass regime, find the hold bar count that maximizes
        average P&L across trades in that regime.

        Uses actual trade data to build a histogram of hold_bars vs pnl
        and returns the mode bar count weighted by P&L.

        Returns dict keyed by BH mass label with optimal_bars and stats.
        """
        bh_labels = PostTradeAnalyzer.BH_MASS_LABELS
        bh_thresholds = PostTradeAnalyzer.BH_MASS_THRESHOLDS

        regime_trades: dict[str, list[PostTradeRecord]] = {lb: [] for lb in bh_labels}
        for t in self.trades:
            lb = _bucket_label(t.bh_mass_at_entry, bh_thresholds, bh_labels)
            regime_trades[lb].append(t)

        result = {}
        for lb, records in regime_trades.items():
            if not records:
                result[lb] = {"optimal_bars": None, "n": 0}
                continue

            # Bin by hold_bars, compute mean P&L per bin
            bar_bins: dict[int, list[float]] = {}
            for t in records:
                bar_bins.setdefault(t.hold_bars, []).append(t.pnl_pct)

            bar_means = {
                bars: (_avg(pnls), len(pnls))
                for bars, pnls in bar_bins.items()
                if len(pnls) >= 2  # require at least 2 samples per bin
            }

            if not bar_means:
                # Fall back to single-sample bins
                bar_means = {
                    bars: (_avg(pnls), len(pnls))
                    for bars, pnls in bar_bins.items()
                }

            if not bar_means:
                result[lb] = {"optimal_bars": None, "n": len(records)}
                continue

            best_bars = max(bar_means, key=lambda b: bar_means[b][0])
            result[lb] = {
                "optimal_bars": best_bars,
                "optimal_avg_pnl_pct": bar_means[best_bars][0],
                "optimal_n": bar_means[best_bars][1],
                "n": len(records),
                "all_bars_stats": {
                    b: {"avg_pnl_pct": v[0], "n": v[1]}
                    for b, v in sorted(bar_means.items())
                },
            }

        return result

    # ------------------------------------------------------------------
    # Per-exit-reason efficiency breakdown
    # ------------------------------------------------------------------

    def exit_reason_efficiency_breakdown(self) -> pd.DataFrame:
        """
        Pivot table: exit_reason x BH mass level, showing avg exit efficiency.
        """
        rows = []
        for t in self.trades:
            lb = _bucket_label(
                t.bh_mass_at_entry,
                PostTradeAnalyzer.BH_MASS_THRESHOLDS,
                PostTradeAnalyzer.BH_MASS_LABELS,
            )
            rows.append({
                "exit_reason": t.exit_reason,
                "bh_mass_level": lb,
                "exit_efficiency": t.exit_efficiency,
            })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df.pivot_table(
            values="exit_efficiency",
            index="exit_reason",
            columns="bh_mass_level",
            aggfunc="mean",
        ).fillna(0.0)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse, json

    parser = argparse.ArgumentParser(description="Post-trade analyzer for LARSA v18")
    parser.add_argument("db", help="Path to trades SQLite database")
    parser.add_argument(
        "--report", default="summary",
        choices=["summary", "entry", "exit", "mfe_mae", "missed", "cohort"],
        help="Which analysis to run",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    analyzer = PostTradeAnalyzer(db_path=args.db)

    if args.report == "summary":
        result = analyzer.summary_report()
    elif args.report == "entry":
        result = analyzer.analyze_entry_quality()
    elif args.report == "exit":
        result = analyzer.analyze_exit_timing()
    elif args.report == "mfe_mae":
        result = analyzer.analyze_mfe_mae()
    elif args.report == "missed":
        result = analyzer.find_missed_opportunities()
    elif args.report == "cohort":
        result = analyzer.cohort_analysis().to_dict(orient="records")
    else:
        result = {}

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        import pprint
        pprint.pprint(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
