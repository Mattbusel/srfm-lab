"""
idea-engine/ingestion/loaders/backtest_loader.py
─────────────────────────────────────────────────
Loads backtest artefacts from tools/backtest_output/ and assembles a
BacktestResult dataclass.

Sources consumed
────────────────
  • crypto_trades.csv      — trade-by-trade log
  • crypto_bh_mc.png       — (metadata only: file size, mtime)
  • *.json                 — any JSON result files in the same directory

Computed metrics
────────────────
  sharpe, cagr, max_dd, win_rate, profit_factor are computed from the
  loaded trades/equity data if not already present in a JSON results file.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import (
    BACKTEST_OUTPUT_DIR,
    CRYPTO_BH_MC_PNG,
    CRYPTO_TRADES_CSV,
)
from ..types import (
    BacktestResult,
    cagr_from_equity,
    max_drawdown,
    safe_float,
    sharpe_from_returns,
)

logger = logging.getLogger(__name__)


# ── constants ─────────────────────────────────────────────────────────────────

# Columns expected in crypto_trades.csv
_CSV_COLS = {
    "exit_time":    "exit_time",
    "sym":          "symbol",
    "entry_price":  "entry_price",
    "exit_price":   "exit_price",
    "dollar_pos":   "dollar_pos",
    "pnl":          "pnl",
    "hold_bars":    "hold_bars",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_trades_csv(path: Path) -> Optional[pd.DataFrame]:
    """
    Load crypto_trades.csv.

    Handles:
    - Missing file (returns None with a warning)
    - Column renaming
    - Datetime parsing
    - Numeric coercion
    """
    if not path.exists():
        logger.warning("Trades CSV not found: %s", path)
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.error("Failed to read %s: %s", path, exc)
        return None

    # Rename known columns
    rename_map = {old: new for old, new in _CSV_COLS.items() if old in df.columns}
    df = df.rename(columns=rename_map)

    # Parse datetime
    if "exit_time" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time"], utc=False, errors="coerce")
        df = df.sort_values("exit_time").reset_index(drop=True)

    # Coerce numeric columns
    for col in ["entry_price", "exit_price", "dollar_pos", "pnl", "hold_bars"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.debug("Loaded %d trades from %s", len(df), path)
    return df


def _build_equity_curve(trades_df: pd.DataFrame, initial_equity: float = 1_000_000.0) -> pd.Series:
    """
    Reconstruct a cumulative equity curve from trade-level PnL.

    Returns a pd.Series indexed by exit_time.
    """
    df = trades_df.copy()
    if "exit_time" not in df.columns or "pnl" not in df.columns:
        return pd.Series(dtype=float)

    df = df.dropna(subset=["exit_time", "pnl"]).sort_values("exit_time")
    equity = initial_equity + df["pnl"].cumsum()
    equity.index = df["exit_time"].values
    equity.name  = "equity"
    return equity


def _compute_win_rate(trades_df: pd.DataFrame) -> Optional[float]:
    if "pnl" not in trades_df.columns:
        return None
    pnl = trades_df["pnl"].dropna()
    if len(pnl) == 0:
        return None
    return float((pnl > 0).sum() / len(pnl))


def _compute_profit_factor(trades_df: pd.DataFrame) -> Optional[float]:
    if "pnl" not in trades_df.columns:
        return None
    pnl = trades_df["pnl"].dropna()
    gross_profit = pnl[pnl > 0].sum()
    gross_loss   = abs(pnl[pnl < 0].sum())
    if gross_loss == 0:
        return None if gross_profit == 0 else float("inf")
    return float(gross_profit / gross_loss)


def _compute_sharpe_from_equity(equity: pd.Series, periods_per_year: int = 8_760) -> Optional[float]:
    """
    Compute Sharpe from equity curve.

    Uses hourly periods by default (crypto 24/7: 8760 hours/year).
    """
    if len(equity) < 2:
        return None
    returns = equity.pct_change().dropna()
    return sharpe_from_returns(returns, periods_per_year=periods_per_year)


def _scan_json_files(directory: Path) -> Dict[str, Any]:
    """
    Scan directory for *.json files and merge their contents.

    Later files override earlier ones on key conflicts.
    Returns a flat merged dict.
    """
    merged: Dict[str, Any] = {}
    for jf in sorted(directory.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                merged.update(data)
                logger.debug("Merged JSON results from %s", jf.name)
            else:
                merged[jf.stem] = data
        except Exception as exc:
            logger.warning("Could not parse %s: %s", jf.name, exc)
    return merged


def _png_metadata(path: Path) -> Dict[str, Any]:
    """Return basic file metadata for a PNG (size, mtime)."""
    if not path.exists():
        return {}
    stat = path.stat()
    return {
        "png_path":  str(path),
        "png_size_bytes": stat.st_size,
        "png_mtime": stat.st_mtime,
    }


def _get_period(trades_df: pd.DataFrame) -> tuple[str, str]:
    if "exit_time" not in trades_df.columns:
        return ("", "")
    ts = pd.to_datetime(trades_df["exit_time"]).dropna()
    if len(ts) == 0:
        return ("", "")
    return (str(ts.min().date()), str(ts.max().date()))


def _get_instruments(trades_df: pd.DataFrame) -> List[str]:
    col = "symbol" if "symbol" in trades_df.columns else "sym" if "sym" in trades_df.columns else None
    if col is None:
        return []
    return sorted(trades_df[col].dropna().unique().tolist())


# ── public API ────────────────────────────────────────────────────────────────

def load_backtest(
    csv_path:   Path = CRYPTO_TRADES_CSV,
    output_dir: Path = BACKTEST_OUTPUT_DIR,
    png_path:   Path = CRYPTO_BH_MC_PNG,
    *,
    initial_equity: float = 1_000_000.0,
    periods_per_year: int = 8_760,
) -> BacktestResult:
    """
    Load all available backtest artefacts and return a BacktestResult.

    Parameters
    ----------
    csv_path        : path to crypto_trades.csv
    output_dir      : directory to scan for JSON result files and PNG
    png_path        : path to crypto_bh_mc.png (metadata only)
    initial_equity  : starting equity for equity curve reconstruction
    periods_per_year: used for Sharpe annualisation (8760 for hourly crypto)

    Returns
    -------
    BacktestResult with all available fields populated.
    """
    logger.info("Loading backtest data from %s", output_dir)

    # 1. Trades CSV
    trades_df = _load_trades_csv(csv_path)

    # 2. JSON results (may override computed metrics)
    json_data = _scan_json_files(output_dir)

    # 3. PNG metadata
    png_meta = _png_metadata(png_path)

    # 4. Build equity curve
    equity_curve: Optional[pd.Series] = None
    if trades_df is not None and "pnl" in trades_df.columns:
        equity_curve = _build_equity_curve(trades_df, initial_equity=initial_equity)
        if len(equity_curve) == 0:
            equity_curve = None

    # 5. Compute metrics — prefer JSON file values, fall back to computed
    def _pick(json_key: str, computed: Optional[float]) -> Optional[float]:
        if json_key in json_data:
            return safe_float(json_data[json_key])
        return computed

    sharpe = _pick(
        "sharpe",
        _compute_sharpe_from_equity(equity_curve, periods_per_year) if equity_curve is not None else None,
    )
    cagr = _pick(
        "cagr",
        cagr_from_equity(equity_curve) if equity_curve is not None else None,
    )
    mdd = _pick(
        "max_dd",
        max_drawdown(equity_curve) if equity_curve is not None else None,
    )
    win_rate = _pick(
        "win_rate",
        _compute_win_rate(trades_df) if trades_df is not None else None,
    )
    pf = _pick(
        "profit_factor",
        _compute_profit_factor(trades_df) if trades_df is not None else None,
    )

    # 6. Period and instruments
    period      = _get_period(trades_df) if trades_df is not None else ("", "")
    instruments = _get_instruments(trades_df) if trades_df is not None else []

    # 7. Extra metadata
    extra: Dict[str, Any] = {**png_meta}
    for k, v in json_data.items():
        if k not in ("sharpe", "cagr", "max_dd", "win_rate", "profit_factor"):
            extra[k] = v

    result = BacktestResult(
        period        = period,
        instruments   = instruments,
        equity_curve  = equity_curve,
        trades_df     = trades_df,
        sharpe        = sharpe,
        cagr          = cagr,
        max_dd        = mdd,
        win_rate      = win_rate,
        profit_factor = pf,
        extra         = extra,
    )

    logger.info(
        "BacktestResult loaded: %d trades | Sharpe=%.3f | CAGR=%.1f%% | MDD=%.1f%%",
        len(trades_df) if trades_df is not None else 0,
        sharpe or float("nan"),
        (cagr or 0) * 100,
        (mdd or 0) * 100,
    )
    return result
