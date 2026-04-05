"""
idea-engine/ingestion/types.py
──────────────────────────────
Shared data types for the ingestion layer.

All dataclasses use slots=False for pickle compatibility with older Python.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ── Enumerations ─────────────────────────────────────────────────────────────

class PatternType(str, Enum):
    TIME_OF_DAY    = "time_of_day"
    REGIME_CLUSTER = "regime_cluster"
    BH_PHYSICS     = "bh_physics"
    DRAWDOWN       = "drawdown"
    CUSTOM         = "custom"


class PatternStatus(str, Enum):
    NEW       = "new"
    CONFIRMED = "confirmed"
    REJECTED  = "rejected"
    PROMOTED  = "promoted"


class EffectSizeType(str, Enum):
    COHENS_D     = "cohens_d"
    CLIFFS_DELTA = "cliffs_delta"
    ETA_SQUARED  = "eta_squared"


class DataSource(str, Enum):
    BACKTEST     = "backtest"
    LIVE         = "live"
    WALK_FORWARD = "walk_forward"
    PAPER        = "paper"


# ── MinedPattern ─────────────────────────────────────────────────────────────

@dataclass
class MinedPattern:
    """
    A statistically interesting pattern discovered by a miner.

    Corresponds to a row in the `patterns` table.
    """
    # Identity
    source:           str                    = ""
    miner:            str                    = ""
    pattern_type:     PatternType            = PatternType.CUSTOM
    label:            str                    = ""
    description:      str                    = ""

    # Feature definition
    feature_dict:     Dict[str, Any]         = field(default_factory=dict)
    window_start:     Optional[str]          = None
    window_end:       Optional[str]          = None
    instruments:      List[str]              = field(default_factory=list)

    # Sample stats
    sample_size:      int                    = 0
    p_value:          Optional[float]        = None
    effect_size:      Optional[float]        = None
    effect_size_type: EffectSizeType         = EffectSizeType.COHENS_D

    # Performance metrics
    win_rate:         Optional[float]        = None
    avg_pnl:          Optional[float]        = None
    avg_pnl_baseline: Optional[float]        = None
    sharpe:           Optional[float]        = None
    max_dd:           Optional[float]        = None
    profit_factor:    Optional[float]        = None

    # Post-filter confidence
    confidence:       Optional[float]        = None

    # Status
    status:           PatternStatus          = PatternStatus.NEW
    tags:             List[str]              = field(default_factory=list)

    # Optional raw data (not persisted to DB directly)
    raw_group:        Optional[pd.Series]    = field(default=None, repr=False)
    raw_baseline:     Optional[pd.Series]    = field(default=None, repr=False)

    def to_db_dict(self) -> dict:
        """Convert to a dict suitable for inserting into the patterns table."""
        return {
            "source":           self.source,
            "miner":            self.miner,
            "pattern_type":     self.pattern_type.value,
            "label":            self.label,
            "description":      self.description,
            "feature_json":     json.dumps(self.feature_dict),
            "window_start":     self.window_start,
            "window_end":       self.window_end,
            "instruments":      json.dumps(self.instruments),
            "sample_size":      self.sample_size,
            "p_value":          self.p_value,
            "effect_size":      self.effect_size,
            "effect_size_type": self.effect_size_type.value,
            "win_rate":         self.win_rate,
            "avg_pnl":          self.avg_pnl,
            "avg_pnl_baseline": self.avg_pnl_baseline,
            "sharpe":           self.sharpe,
            "max_dd":           self.max_dd,
            "profit_factor":    self.profit_factor,
            "confidence":       self.confidence,
            "status":           self.status.value,
            "tags":             ",".join(self.tags),
        }


# ── BacktestResult ────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """
    Loaded result from a backtest run.

    Attributes
    ----------
    period          : (start_date, end_date) strings
    instruments     : list of traded symbols
    equity_curve    : pd.Series indexed by datetime, values = portfolio equity
    trades_df       : full trades DataFrame
    sharpe          : annualised Sharpe ratio
    cagr            : compounded annual growth rate (fraction, e.g. 0.25 = 25 %)
    max_dd          : max drawdown (fraction, negative, e.g. -0.12)
    win_rate        : fraction of profitable trades
    profit_factor   : gross profit / gross loss
    extra           : arbitrary additional metadata from source files
    """
    period:        tuple[str, str]               = ("", "")
    instruments:   List[str]                     = field(default_factory=list)
    equity_curve:  Optional[pd.Series]           = field(default=None, repr=False)
    trades_df:     Optional[pd.DataFrame]        = field(default=None, repr=False)
    sharpe:        Optional[float]               = None
    cagr:          Optional[float]               = None
    max_dd:        Optional[float]               = None
    win_rate:      Optional[float]               = None
    profit_factor: Optional[float]               = None
    extra:         Dict[str, Any]                = field(default_factory=dict)


# ── LiveTradeData ─────────────────────────────────────────────────────────────

@dataclass
class LiveTradeData:
    """
    Loaded data from live_trades.db.

    Attributes
    ----------
    trades              : trades table as DataFrame
    equity_snapshots    : equity_snapshots table as DataFrame
    positions           : positions table as DataFrame
    regime_log          : regime_log table as DataFrame
    equity_series       : normalised equity curve (pd.Series indexed by datetime)
    """
    trades:           Optional[pd.DataFrame]  = field(default=None, repr=False)
    equity_snapshots: Optional[pd.DataFrame]  = field(default=None, repr=False)
    positions:        Optional[pd.DataFrame]  = field(default=None, repr=False)
    regime_log:       Optional[pd.DataFrame]  = field(default=None, repr=False)
    equity_series:    Optional[pd.Series]     = field(default=None, repr=False)


# ── WalkForwardResult ─────────────────────────────────────────────────────────

@dataclass
class FoldMetrics:
    """Metrics for a single walk-forward fold."""
    fold_id:    int
    is_start:   Optional[str]  = None
    is_end:     Optional[str]  = None
    oos_start:  Optional[str]  = None
    oos_end:    Optional[str]  = None
    is_sharpe:  Optional[float] = None
    oos_sharpe: Optional[float] = None
    is_cagr:    Optional[float] = None
    oos_cagr:   Optional[float] = None
    is_dd:      Optional[float] = None
    oos_dd:     Optional[float] = None
    is_wr:      Optional[float] = None
    oos_wr:     Optional[float] = None
    params:     Dict[str, Any]  = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """
    Aggregated walk-forward analysis results.

    Attributes
    ----------
    source_dir          : directory that was scanned
    folds               : list of FoldMetrics, one per fold
    mean_is_sharpe      : mean Sharpe over IS folds
    mean_oos_sharpe     : mean Sharpe over OOS folds
    oos_degradation     : (mean_oos_sharpe - mean_is_sharpe) / abs(mean_is_sharpe)
    is_equity_curves    : list of pd.Series, one per IS fold
    oos_equity_curves   : list of pd.Series, one per OOS fold
    extra               : additional metadata
    """
    source_dir:         str                        = ""
    folds:              List[FoldMetrics]          = field(default_factory=list)
    mean_is_sharpe:     Optional[float]            = None
    mean_oos_sharpe:    Optional[float]            = None
    oos_degradation:    Optional[float]            = None
    is_equity_curves:   List[pd.Series]            = field(default_factory=list, repr=False)
    oos_equity_curves:  List[pd.Series]            = field(default_factory=list, repr=False)
    extra:              Dict[str, Any]             = field(default_factory=dict)


# ── utility helpers ───────────────────────────────────────────────────────────

def safe_float(v: Any) -> Optional[float]:
    """Convert to float, returning None on failure."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f) else f  # catch NaN
    except (TypeError, ValueError):
        return None


def sharpe_from_returns(returns: pd.Series, periods_per_year: int = 252) -> Optional[float]:
    """Annualised Sharpe from a returns series."""
    if returns is None or len(returns) < 2:
        return None
    mu  = returns.mean()
    std = returns.std()
    if std == 0 or np.isnan(std):
        return None
    return float(mu / std * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> Optional[float]:
    """Maximum drawdown as a negative fraction."""
    if equity is None or len(equity) < 2:
        return None
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min())


def cagr_from_equity(equity: pd.Series) -> Optional[float]:
    """CAGR from an equity curve.  Assumes daily bars."""
    if equity is None or len(equity) < 2:
        return None
    start = equity.iloc[0]
    end   = equity.iloc[-1]
    if start <= 0:
        return None
    n_years = len(equity) / 252.0
    if n_years <= 0:
        return None
    return float((end / start) ** (1.0 / n_years) - 1.0)
