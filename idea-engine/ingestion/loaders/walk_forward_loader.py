"""
idea-engine/ingestion/loaders/walk_forward_loader.py
─────────────────────────────────────────────────────
Scans research/walk_forward/ for result artefacts and assembles a
WalkForwardResult dataclass.

Files consumed (in order of preference)
────────────────────────────────────────
  1. walk_forward_results.json    — structured WF summary
  2. *.json                       — other JSON result files
  3. *fold*.csv / *oos*.csv       — per-fold equity / trade CSVs
  4. *is*.csv                     — IS fold CSVs

IS / OOS split performance is extracted from per-fold data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import WALK_FORWARD_DIR
from ..types import (
    FoldMetrics,
    WalkForwardResult,
    cagr_from_equity,
    max_drawdown,
    safe_float,
    sharpe_from_returns,
)

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_json_results(directory: Path) -> Dict[str, Any]:
    """Merge all JSON files in directory, prioritising walk_forward_results.json."""
    merged: Dict[str, Any] = {}
    priority_file = directory / "walk_forward_results.json"
    candidates    = sorted(directory.glob("*.json"))
    # Move priority file to the end so it overwrites others
    if priority_file in candidates:
        candidates.remove(priority_file)
        candidates.append(priority_file)
    for jf in candidates:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                merged.update(data)
                logger.debug("Loaded JSON: %s", jf.name)
            elif isinstance(data, list):
                merged[jf.stem] = data
        except Exception as exc:
            logger.warning("Could not parse %s: %s", jf.name, exc)
    return merged


def _sharpe_from_df(df: pd.DataFrame, equity_col: str = "equity") -> Optional[float]:
    if equity_col not in df.columns or len(df) < 3:
        return None
    eq = df[equity_col].dropna()
    if len(eq) < 3:
        return None
    returns = eq.pct_change().dropna()
    return sharpe_from_returns(returns, periods_per_year=252)


def _cagr_from_df(df: pd.DataFrame, equity_col: str = "equity") -> Optional[float]:
    if equity_col not in df.columns:
        return None
    return cagr_from_equity(df[equity_col].dropna())


def _mdd_from_df(df: pd.DataFrame, equity_col: str = "equity") -> Optional[float]:
    if equity_col not in df.columns:
        return None
    return max_drawdown(df[equity_col].dropna())


def _win_rate_from_df(df: pd.DataFrame) -> Optional[float]:
    if "pnl" not in df.columns:
        return None
    pnl = pd.to_numeric(df["pnl"], errors="coerce").dropna()
    if len(pnl) == 0:
        return None
    return float((pnl > 0).sum() / len(pnl))


def _detect_equity_col(df: pd.DataFrame) -> str:
    for c in ["equity", "cumulative_equity", "portfolio_value", "capital", "nav"]:
        if c in df.columns:
            return c
    return "equity"


def _load_csv_as_series(path: Path) -> Optional[pd.Series]:
    """Load a CSV and return the equity series."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None
    col = _detect_equity_col(df)
    if col not in df.columns:
        # Try first numeric column
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return None
        col = num_cols[0]
    # Detect index column
    for ts_col in ["ts", "timestamp", "date", "datetime", "time"]:
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            s = df.set_index(ts_col)[col].dropna()
            s.name = "equity"
            return s
    s = df[col].dropna()
    s.name = "equity"
    return s


# ── fold file scanning ────────────────────────────────────────────────────────

def _scan_fold_csvs(directory: Path) -> Dict[str, List[Path]]:
    """
    Scan for CSV files with 'fold', 'oos', 'is' in their names.

    Returns dict with keys 'is' and 'oos' mapping to sorted lists of paths.
    """
    all_csvs = sorted(directory.rglob("*.csv"))
    is_files  = [p for p in all_csvs if any(tok in p.stem.lower() for tok in ["_is_", "is_fold", "in_sample"])]
    oos_files = [p for p in all_csvs if any(tok in p.stem.lower() for tok in ["_oos_", "oos_fold", "out_of_sample", "out_sample"])]
    fold_files = [p for p in all_csvs if "fold" in p.stem.lower() and p not in is_files + oos_files]
    return {"is": is_files, "oos": oos_files, "fold": fold_files}


def _extract_fold_id(path: Path) -> int:
    """Try to extract a fold number from a filename, e.g. fold_03_oos.csv → 3."""
    import re
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def _build_folds_from_json(data: Dict[str, Any]) -> List[FoldMetrics]:
    """Parse fold list from JSON data if available."""
    folds: List[FoldMetrics] = []
    raw_folds = data.get("folds") or data.get("fold_results") or data.get("splits") or []
    if not isinstance(raw_folds, list):
        return folds
    for i, f in enumerate(raw_folds):
        if not isinstance(f, dict):
            continue
        fm = FoldMetrics(
            fold_id    = f.get("fold_id", i),
            is_start   = str(f.get("is_start", "")) or None,
            is_end     = str(f.get("is_end", "")) or None,
            oos_start  = str(f.get("oos_start", "")) or None,
            oos_end    = str(f.get("oos_end", "")) or None,
            is_sharpe  = safe_float(f.get("is_sharpe")),
            oos_sharpe = safe_float(f.get("oos_sharpe")),
            is_cagr    = safe_float(f.get("is_cagr")),
            oos_cagr   = safe_float(f.get("oos_cagr")),
            is_dd      = safe_float(f.get("is_max_dd") or f.get("is_dd")),
            oos_dd     = safe_float(f.get("oos_max_dd") or f.get("oos_dd")),
            is_wr      = safe_float(f.get("is_win_rate") or f.get("is_wr")),
            oos_wr     = safe_float(f.get("oos_win_rate") or f.get("oos_wr")),
            params     = f.get("params", {}),
        )
        folds.append(fm)
    return folds


def _build_folds_from_csvs(fold_map: Dict[str, List[Path]]) -> tuple[List[FoldMetrics], List[pd.Series], List[pd.Series]]:
    """Build FoldMetrics by pairing IS/OOS CSV files."""
    is_curves:  List[pd.Series] = []
    oos_curves: List[pd.Series] = []
    folds:      List[FoldMetrics] = []

    # Load individual IS curves
    for p in fold_map["is"]:
        s = _load_csv_as_series(p)
        if s is not None:
            is_curves.append(s)

    # Load individual OOS curves
    for p in fold_map["oos"]:
        s = _load_csv_as_series(p)
        if s is not None:
            oos_curves.append(s)

    # Build paired FoldMetrics
    n_folds = max(len(is_curves), len(oos_curves))
    for i in range(n_folds):
        is_s  = is_curves[i]  if i < len(is_curves)  else None
        oos_s = oos_curves[i] if i < len(oos_curves) else None

        def _sharpe(s: Optional[pd.Series]) -> Optional[float]:
            if s is None or len(s) < 3:
                return None
            r = s.pct_change().dropna()
            return sharpe_from_returns(r, 252)

        folds.append(FoldMetrics(
            fold_id    = i,
            is_sharpe  = _sharpe(is_s),
            oos_sharpe = _sharpe(oos_s),
            is_cagr    = cagr_from_equity(is_s) if is_s is not None else None,
            oos_cagr   = cagr_from_equity(oos_s) if oos_s is not None else None,
            is_dd      = max_drawdown(is_s) if is_s is not None else None,
            oos_dd     = max_drawdown(oos_s) if oos_s is not None else None,
        ))

    return folds, is_curves, oos_curves


# ── public API ────────────────────────────────────────────────────────────────

def load_walk_forward(directory: Path = WALK_FORWARD_DIR) -> WalkForwardResult:
    """
    Load walk-forward results from the given directory.

    Strategy
    --------
    1. Scan JSON files for structured fold data.
    2. Scan CSV files for IS/OOS equity curves.
    3. Merge and compute aggregate IS vs OOS degradation.

    Returns
    -------
    WalkForwardResult
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning("Walk-forward directory not found: %s", directory)
        return WalkForwardResult(source_dir=str(directory))

    logger.info("Loading walk-forward results from %s", directory)

    # Step 1: JSON
    json_data = _load_json_results(directory)

    # Step 2: fold CSVs
    fold_map = _scan_fold_csvs(directory)

    # Step 3: build folds
    folds_from_json = _build_folds_from_json(json_data)
    folds_from_csv, is_curves, oos_curves = _build_folds_from_csvs(fold_map)

    # Prefer JSON folds if available
    folds: List[FoldMetrics] = folds_from_json if folds_from_json else folds_from_csv

    # Step 4: top-level Sharpe from JSON or computed from folds
    mean_is_sharpe  = safe_float(json_data.get("mean_is_sharpe")  or json_data.get("is_sharpe"))
    mean_oos_sharpe = safe_float(json_data.get("mean_oos_sharpe") or json_data.get("oos_sharpe"))

    if mean_is_sharpe is None and folds:
        is_vals = [f.is_sharpe for f in folds if f.is_sharpe is not None]
        mean_is_sharpe = float(np.mean(is_vals)) if is_vals else None

    if mean_oos_sharpe is None and folds:
        oos_vals = [f.oos_sharpe for f in folds if f.oos_sharpe is not None]
        mean_oos_sharpe = float(np.mean(oos_vals)) if oos_vals else None

    # OOS degradation: (oos - is) / |is|
    oos_degradation: Optional[float] = None
    if mean_is_sharpe is not None and mean_oos_sharpe is not None and mean_is_sharpe != 0:
        oos_degradation = (mean_oos_sharpe - mean_is_sharpe) / abs(mean_is_sharpe)

    extra: Dict[str, Any] = {
        k: v for k, v in json_data.items()
        if k not in ("folds", "fold_results", "splits", "mean_is_sharpe", "mean_oos_sharpe", "is_sharpe", "oos_sharpe")
    }

    result = WalkForwardResult(
        source_dir        = str(directory),
        folds             = folds,
        mean_is_sharpe    = mean_is_sharpe,
        mean_oos_sharpe   = mean_oos_sharpe,
        oos_degradation   = oos_degradation,
        is_equity_curves  = is_curves,
        oos_equity_curves = oos_curves,
        extra             = extra,
    )

    logger.info(
        "WalkForwardResult: %d folds | IS Sharpe=%.3f | OOS Sharpe=%.3f | Degradation=%.1f%%",
        len(folds),
        mean_is_sharpe or float("nan"),
        mean_oos_sharpe or float("nan"),
        (oos_degradation or 0) * 100,
    )
    return result
