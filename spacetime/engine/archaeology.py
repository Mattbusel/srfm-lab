"""
archaeology.py — Trade archaeology for Spacetime Arena.

Loads QC trade CSV files, reconstructs BH state at each trade entry,
stores results in SQLite, and provides query functions.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from sqlalchemy import (
    Column, Float, Integer, String, Text, create_engine, func, select, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "db" / "trades.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

ENGINE = create_engine(f"sqlite:///{DB_PATH}", future=True)


# ---------------------------------------------------------------------------
# ORM
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class Trade(Base):
    __tablename__ = "trades"

    id                : int   = Column(Integer, primary_key=True, autoincrement=True)
    run_name          : str   = Column(String(128), index=True)
    sym               : str   = Column(String(32), index=True)
    entry_time        : str   = Column(String(32))
    exit_time         : str   = Column(String(32))
    entry_price       : float = Column(Float)
    exit_price        : float = Column(Float)
    pnl               : float = Column(Float)
    hold_hours        : float = Column(Float)
    tf_score          : int   = Column(Integer, index=True)
    regime            : str   = Column(String(32), index=True)
    bh_mass_at_entry  : float = Column(Float)
    pos_floor_at_entry: float = Column(Float)
    mfe               : float = Column(Float)
    mae               : float = Column(Float)


Base.metadata.create_all(ENGINE)
SessionLocal = sessionmaker(bind=ENGINE, expire_on_commit=False)


# ---------------------------------------------------------------------------
# QC CSV parsing
# ---------------------------------------------------------------------------

QC_COLUMN_MAP = {
    "Entry Time":  "entry_time",
    "Exit Time":   "exit_time",
    "Symbol":      "sym",
    "Symbols":     "sym",
    "Profit":      "pnl",
    "PnL":         "pnl",
    "Entry Price": "entry_price",
    "Exit Price":  "exit_price",
}


def parse_qc_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Parse a QuantConnect trade CSV file.

    Handles column name variations from different QC report formats.
    Returns a normalized DataFrame with columns:
      sym, entry_time, exit_time, entry_price, exit_price, pnl
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df = df.rename(columns={c: QC_COLUMN_MAP[c] for c in df.columns if c in QC_COLUMN_MAP})

    # Fallbacks
    if "sym" not in df.columns:
        for c in df.columns:
            if c.lower() in ("symbol", "ticker", "instrument"):
                df = df.rename(columns={c: "sym"})
                break

    if "pnl" not in df.columns:
        for c in df.columns:
            if c.lower() in ("profit", "pnl", "return", "net profit", "net_profit"):
                df = df.rename(columns={c: "pnl"})
                break

    required = {"sym", "entry_time", "exit_time", "entry_price", "exit_price", "pnl"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"QC CSV missing columns: {missing}. Available: {list(df.columns)}")

    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    df["exit_price"]  = pd.to_numeric(df["exit_price"],  errors="coerce")
    df["pnl"]         = pd.to_numeric(df["pnl"],         errors="coerce")

    df = df.dropna(subset=["entry_time", "entry_price", "pnl"])
    return df


# ---------------------------------------------------------------------------
# BH state reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_bh_state(
    sym: str,
    entry_time: str,
    price_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Replay BH engine up to entry_time on price_df and return state at that point.
    Returns dict with tf_score, regime, bh_mass_at_entry, pos_floor_at_entry.
    """
    if price_df is None or price_df.empty:
        return {"tf_score": 0, "regime": "UNKNOWN", "bh_mass_at_entry": 0.0, "pos_floor_at_entry": 0.0}

    try:
        import sys
        _LIB = Path(__file__).parent.parent.parent / "lib"
        sys.path.insert(0, str(_LIB))
        from .bh_engine import BHEngine

        # Filter price data up to entry_time
        entry_ts = pd.Timestamp(entry_time)
        df_sub   = price_df[price_df.index <= entry_ts].copy()

        if len(df_sub) < 20:
            return {"tf_score": 0, "regime": "UNKNOWN", "bh_mass_at_entry": 0.0, "pos_floor_at_entry": 0.0}

        engine = BHEngine(sym, long_only=True)
        result = engine.run(df_sub)

        if result.bar_states:
            last_state = result.bar_states[-1]
            return {
                "tf_score":           last_state.tf_score,
                "regime":             last_state.regime,
                "bh_mass_at_entry":   last_state.bh_mass_1d,
                "pos_floor_at_entry": last_state.pos_floor,
            }
    except Exception as e:
        logger.warning("BH state reconstruction failed for %s at %s: %s", sym, entry_time, e)

    return {"tf_score": 0, "regime": "UNKNOWN", "bh_mass_at_entry": 0.0, "pos_floor_at_entry": 0.0}


# ---------------------------------------------------------------------------
# Main archaeology function
# ---------------------------------------------------------------------------

def run_archaeology(
    csv_path: str | Path,
    run_name: str,
    price_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> List[Trade]:
    """
    Parse QC CSV, reconstruct BH state at each trade entry, store in DB.

    Parameters
    ----------
    csv_path   : path to QC trade CSV
    run_name   : label for this run (stored in DB)
    price_data : {sym: OHLCV DataFrame} — used for BH state reconstruction
                 If None, BH state fields will be 0/UNKNOWN.
    """
    df = parse_qc_csv(csv_path)
    logger.info("Parsed %d trades from %s", len(df), csv_path)

    trades_to_insert: List[Trade] = []

    for _, row in df.iterrows():
        sym        = str(row.get("sym", ""))
        entry_time = str(row.get("entry_time", ""))
        exit_time  = str(row.get("exit_time", ""))
        entry_px   = float(row.get("entry_price", 0.0))
        exit_px    = float(row.get("exit_price", 0.0))
        pnl        = float(row.get("pnl", 0.0))

        # Hold hours
        try:
            hold_hours = (
                pd.Timestamp(exit_time) - pd.Timestamp(entry_time)
            ).total_seconds() / 3600
        except Exception:
            hold_hours = 0.0

        # MFE / MAE (from CSV if present, else estimate)
        mfe = float(row.get("mfe", (exit_px - entry_px) / max(entry_px, 1e-9)))
        mae = float(row.get("mae", 0.0))

        # BH state reconstruction
        px_df = (price_data or {}).get(sym)
        bh_state = _reconstruct_bh_state(sym, entry_time, px_df)

        trades_to_insert.append(Trade(
            run_name=run_name,
            sym=sym,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_px,
            exit_price=exit_px,
            pnl=pnl,
            hold_hours=hold_hours,
            tf_score=bh_state["tf_score"],
            regime=bh_state["regime"],
            bh_mass_at_entry=bh_state["bh_mass_at_entry"],
            pos_floor_at_entry=bh_state["pos_floor_at_entry"],
            mfe=mfe,
            mae=mae,
        ))

    with SessionLocal() as session:
        session.add_all(trades_to_insert)
        session.commit()

    logger.info("Inserted %d trades for run '%s'", len(trades_to_insert), run_name)
    return trades_to_insert


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def get_by_regime(
    regime: str,
    run_name: Optional[str] = None,
    sym: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return all trades matching the given regime."""
    with SessionLocal() as session:
        q = select(Trade).where(Trade.regime == regime.upper())
        if run_name:
            q = q.where(Trade.run_name == run_name)
        if sym:
            q = q.where(Trade.sym == sym)
        rows = session.scalars(q).all()
        return [_trade_to_dict(r) for r in rows]


def get_by_tf_score(
    min_score: int = 4,
    max_score: int = 7,
    run_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return all trades with tf_score in [min_score, max_score]."""
    with SessionLocal() as session:
        q = (
            select(Trade)
            .where(Trade.tf_score >= min_score)
            .where(Trade.tf_score <= max_score)
        )
        if run_name:
            q = q.where(Trade.run_name == run_name)
        rows = session.scalars(q).all()
        return [_trade_to_dict(r) for r in rows]


def get_edge_by_tf_score(run_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Win rate and profit factor grouped by tf_score.
    Returns list of {tf_score, trade_count, win_rate, profit_factor, avg_pnl}.
    """
    with SessionLocal() as session:
        q = select(Trade)
        if run_name:
            q = q.where(Trade.run_name == run_name)
        all_trades = session.scalars(q).all()

    by_tf: Dict[int, List[float]] = {}
    for t in all_trades:
        by_tf.setdefault(t.tf_score, []).append(t.pnl)

    result = []
    for score, pnls in sorted(by_tf.items()):
        import numpy as np
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        pf     = sum(wins) / (abs(sum(losses)) + 1e-9) if losses else float("inf")
        result.append({
            "tf_score":      score,
            "trade_count":   len(pnls),
            "win_rate":      len(wins) / len(pnls) if pnls else 0.0,
            "profit_factor": round(min(pf, 99.9), 3),
            "avg_pnl":       float(np.mean(pnls)),
        })
    return result


def get_optimal_decay_rate(
    sym: str,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Find the pos_floor decay rate that best explains actual trade outcomes.
    Proxy: group by pos_floor_at_entry buckets, find the bucket with best profit_factor.
    """
    with SessionLocal() as session:
        q = select(Trade).where(Trade.sym == sym)
        if run_name:
            q = q.where(Trade.run_name == run_name)
        trades = session.scalars(q).all()

    if not trades:
        return {"sym": sym, "optimal_decay": 0.95, "note": "no trades found"}

    import numpy as np

    # Bucket pos_floor_at_entry into quartiles
    floors = np.array([t.pos_floor_at_entry for t in trades])
    pnls   = np.array([t.pnl for t in trades])

    if floors.max() == floors.min():
        return {"sym": sym, "optimal_decay": 0.95, "note": "insufficient variance in pos_floor"}

    buckets  = np.percentile(floors, [0, 25, 50, 75, 100])
    best_pf  = -1.0
    best_bucket_idx = 0

    for i in range(len(buckets) - 1):
        mask = (floors >= buckets[i]) & (floors < buckets[i + 1])
        bucket_pnls = pnls[mask]
        if len(bucket_pnls) < 3:
            continue
        wins   = bucket_pnls[bucket_pnls > 0]
        losses = bucket_pnls[bucket_pnls <= 0]
        pf     = float(wins.sum()) / (abs(float(losses.sum())) + 1e-9)
        if pf > best_pf:
            best_pf = pf
            best_bucket_idx = i

    # Map bucket index to approximate decay rate
    # Higher pos_floor (less decay) corresponds to decay closer to 0.99
    # Lower pos_floor (more decay) → decay closer to 0.90
    decay_map = [0.90, 0.93, 0.95, 0.97]
    optimal_decay = decay_map[min(best_bucket_idx, len(decay_map) - 1)]

    return {
        "sym":           sym,
        "optimal_decay": optimal_decay,
        "best_pf":       round(best_pf, 3),
        "bucket_idx":    best_bucket_idx,
        "note":          "approximate optimal pos_floor decay based on trade outcomes",
    }


def _trade_to_dict(t: Trade) -> Dict[str, Any]:
    return {
        "id":                 t.id,
        "run_name":           t.run_name,
        "sym":                t.sym,
        "entry_time":         t.entry_time,
        "exit_time":          t.exit_time,
        "entry_price":        t.entry_price,
        "exit_price":         t.exit_price,
        "pnl":                t.pnl,
        "hold_hours":         t.hold_hours,
        "tf_score":           t.tf_score,
        "regime":             t.regime,
        "bh_mass_at_entry":   t.bh_mass_at_entry,
        "pos_floor_at_entry": t.pos_floor_at_entry,
        "mfe":                t.mfe,
        "mae":                t.mae,
    }


def get_trades(
    sym: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    regime: Optional[str] = None,
    min_tf_score: Optional[int] = None,
    run_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generic trade query with filters."""
    with SessionLocal() as session:
        q = select(Trade)
        if sym:
            q = q.where(Trade.sym == sym)
        if from_date:
            q = q.where(Trade.entry_time >= from_date)
        if to_date:
            q = q.where(Trade.entry_time <= to_date)
        if regime:
            q = q.where(Trade.regime == regime.upper())
        if min_tf_score is not None:
            q = q.where(Trade.tf_score >= min_tf_score)
        if run_name:
            q = q.where(Trade.run_name == run_name)
        rows = session.scalars(q).all()
        return [_trade_to_dict(r) for r in rows]
