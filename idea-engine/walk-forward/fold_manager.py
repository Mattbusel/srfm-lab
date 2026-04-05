"""
walk-forward/fold_manager.py
─────────────────────────────
Fold management for Walk-Forward Analysis.

Two fold modes are supported:

  * ANCHORED — the in-sample window expands with each fold (the IS start
    is always the beginning of the dataset).  This is the standard WFA
    discipline used for strategy validation.

  * ROLLING  — both IS and OOS windows are fixed-size and slide forward
    together.  Useful for detecting regime drift and time-varying behaviour.

Key invariant enforced throughout: OOS start is always strictly after IS end,
so there is zero lookahead contamination.

Usage
-----
    from fold_manager import FoldManager, FoldMode

    fm = FoldManager()
    folds = fm.create_folds(total_bars=5_000, n_folds=8,
                             in_sample_ratio=0.75, mode=FoldMode.ANCHORED)
    for fold in folds:
        is_start, is_end, oos_start, oos_end = fm.get_date_range(fold)
        is_df, oos_df = fm.split_data(full_df, fold)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Enumerations ──────────────────────────────────────────────────────────────

class FoldMode(str, Enum):
    """Walk-forward fold construction mode."""
    ANCHORED = "anchored"   # expanding IS, fixed OOS
    ROLLING  = "rolling"    # fixed IS + OOS windows slide forward


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class AnchoredFold:
    """
    A single anchored (expanding in-sample) walk-forward fold.

    Attributes
    ----------
    fold_number   : 1-based fold index
    is_start_bar  : integer bar index at start of IS window (always 0 for anchored)
    is_end_bar    : integer bar index at end of IS window (inclusive)
    oos_start_bar : integer bar index at start of OOS window
    oos_end_bar   : integer bar index at end of OOS window (inclusive)
    is_start_date : ISO-8601 string; populated by get_date_range if index is datetime
    is_end_date   : ISO-8601 string
    oos_start_date: ISO-8601 string
    oos_end_date  : ISO-8601 string
    mode          : always FoldMode.ANCHORED
    """
    fold_number:    int
    is_start_bar:   int
    is_end_bar:     int
    oos_start_bar:  int
    oos_end_bar:    int
    is_start_date:  str = ""
    is_end_date:    str = ""
    oos_start_date: str = ""
    oos_end_date:   str = ""
    mode:           FoldMode = FoldMode.ANCHORED

    @property
    def is_bars(self) -> int:
        """Number of bars in the IS window."""
        return self.is_end_bar - self.is_start_bar + 1

    @property
    def oos_bars(self) -> int:
        """Number of bars in the OOS window."""
        return self.oos_end_bar - self.oos_start_bar + 1

    def validate(self) -> None:
        """Raise ValueError if lookahead bias is detected."""
        if self.oos_start_bar <= self.is_end_bar:
            raise ValueError(
                f"Fold {self.fold_number}: OOS start bar ({self.oos_start_bar}) must "
                f"be strictly greater than IS end bar ({self.is_end_bar}).  "
                "Lookahead bias detected!"
            )
        if self.is_start_bar < 0:
            raise ValueError(f"Fold {self.fold_number}: is_start_bar must be >= 0.")
        if self.oos_start_bar > self.oos_end_bar:
            raise ValueError(f"Fold {self.fold_number}: oos_start_bar > oos_end_bar.")


@dataclass
class RollingFold:
    """
    A single rolling (fixed-size) walk-forward fold.

    Attributes
    ----------
    fold_number   : 1-based fold index
    is_start_bar  : integer bar index at start of IS window
    is_end_bar    : integer bar index at end of IS window (inclusive)
    oos_start_bar : integer bar index at start of OOS window
    oos_end_bar   : integer bar index at end of OOS window (inclusive)
    is_start_date : ISO-8601 string
    is_end_date   : ISO-8601 string
    oos_start_date: ISO-8601 string
    oos_end_date  : ISO-8601 string
    mode          : always FoldMode.ROLLING
    """
    fold_number:    int
    is_start_bar:   int
    is_end_bar:     int
    oos_start_bar:  int
    oos_end_bar:    int
    is_start_date:  str = ""
    is_end_date:    str = ""
    oos_start_date: str = ""
    oos_end_date:   str = ""
    mode:           FoldMode = FoldMode.ROLLING

    @property
    def is_bars(self) -> int:
        return self.is_end_bar - self.is_start_bar + 1

    @property
    def oos_bars(self) -> int:
        return self.oos_end_bar - self.oos_start_bar + 1

    def validate(self) -> None:
        """Raise ValueError if lookahead bias is detected."""
        if self.oos_start_bar <= self.is_end_bar:
            raise ValueError(
                f"Fold {self.fold_number}: OOS start bar ({self.oos_start_bar}) must "
                f"be strictly greater than IS end bar ({self.is_end_bar}).  "
                "Lookahead bias detected!"
            )
        if self.is_start_bar > self.is_end_bar:
            raise ValueError(f"Fold {self.fold_number}: is_start_bar > is_end_bar.")
        if self.oos_start_bar > self.oos_end_bar:
            raise ValueError(f"Fold {self.fold_number}: oos_start_bar > oos_end_bar.")


# Type alias for either fold kind
AnyFold = Union[AnchoredFold, RollingFold]


# ── FoldManager ───────────────────────────────────────────────────────────────

class FoldManager:
    """
    Creates and manages walk-forward analysis folds.

    Parameters
    ----------
    gap_bars : int
        Minimum number of bars gap between IS end and OOS start.
        Defaults to 0 (bars are contiguous).  A positive value can
        model trading latency / execution delay.
    """

    def __init__(self, gap_bars: int = 0) -> None:
        self.gap_bars = max(0, int(gap_bars))

    # ------------------------------------------------------------------
    # Primary factory
    # ------------------------------------------------------------------

    def create_folds(
        self,
        total_bars: int,
        n_folds: int = 8,
        in_sample_ratio: float = 0.75,
        mode: FoldMode = FoldMode.ANCHORED,
    ) -> List[AnyFold]:
        """
        Create a list of walk-forward folds.

        Parameters
        ----------
        total_bars      : total number of bars in the dataset
        n_folds         : number of folds to create
        in_sample_ratio : fraction of each fold devoted to in-sample data
                          (ignored for ANCHORED mode — IS grows over time)
        mode            : ANCHORED or ROLLING

        Returns
        -------
        List of AnchoredFold or RollingFold instances, validated for no lookahead.
        """
        if total_bars < 2:
            raise ValueError("total_bars must be >= 2.")
        if n_folds < 1:
            raise ValueError("n_folds must be >= 1.")
        if not (0.0 < in_sample_ratio < 1.0):
            raise ValueError("in_sample_ratio must be strictly between 0 and 1.")

        if mode == FoldMode.ANCHORED:
            folds = self._create_anchored_folds(total_bars, n_folds)
        else:
            folds = self._create_rolling_folds(total_bars, n_folds, in_sample_ratio)

        # Validate every fold
        for fold in folds:
            fold.validate()

        logger.debug(
            "Created %d %s folds over %d bars (gap=%d).",
            len(folds), mode.value, total_bars, self.gap_bars,
        )
        return folds

    # ------------------------------------------------------------------
    # Anchored fold construction
    # ------------------------------------------------------------------

    def _create_anchored_folds(
        self,
        total_bars: int,
        n_folds: int,
    ) -> List[AnchoredFold]:
        """
        Build anchored (expanding IS) folds.

        The OOS window is of fixed size = total_bars / (n_folds + 1).
        IS always starts at bar 0 and expands by one OOS width per fold.

        Layout example (n_folds=4, total_bars=100):
            OOS window ≈ 20 bars each
            Fold 1:  IS=[0..19]   OOS=[20..39]
            Fold 2:  IS=[0..39]   OOS=[40..59]
            Fold 3:  IS=[0..59]   OOS=[60..79]
            Fold 4:  IS=[0..79]   OOS=[80..99]
        """
        oos_width = max(1, total_bars // (n_folds + 1))
        folds: List[AnchoredFold] = []

        for i in range(n_folds):
            oos_start = oos_width * (i + 1) + self.gap_bars
            oos_end   = min(oos_start + oos_width - 1, total_bars - 1)
            is_start  = 0
            is_end    = oos_start - 1 - self.gap_bars

            # Safety: skip degenerate folds
            if is_end < is_start or oos_start > total_bars - 1:
                logger.debug("Skipping degenerate anchored fold %d.", i + 1)
                continue

            folds.append(AnchoredFold(
                fold_number    = i + 1,
                is_start_bar   = is_start,
                is_end_bar     = is_end,
                oos_start_bar  = oos_start,
                oos_end_bar    = oos_end,
                mode           = FoldMode.ANCHORED,
            ))

        return folds

    # ------------------------------------------------------------------
    # Rolling fold construction
    # ------------------------------------------------------------------

    def _create_rolling_folds(
        self,
        total_bars: int,
        n_folds: int,
        in_sample_ratio: float,
    ) -> List[RollingFold]:
        """
        Build rolling (fixed-size IS+OOS) folds.

        The combined IS+OOS window slides forward by OOS_width each step.

        Parameters
        ----------
        total_bars      : total bars available
        n_folds         : desired number of folds
        in_sample_ratio : IS fraction of combined window
        """
        # Combined window size such that n_folds windows fit
        # combined * n_folds + IS_bars ≤ total_bars
        # Solve approximately:
        # Step size = OOS size.  total = IS + n_folds * OOS
        # with IS / (IS + OOS) = in_sample_ratio
        oos_ratio = 1.0 - in_sample_ratio
        step = max(1, int(total_bars / (n_folds + in_sample_ratio / oos_ratio)))
        is_size = max(1, int(step * in_sample_ratio / oos_ratio))

        folds: List[RollingFold] = []
        for i in range(n_folds):
            is_start  = i * step
            is_end    = is_start + is_size - 1
            oos_start = is_end + 1 + self.gap_bars
            oos_end   = oos_start + step - 1

            # Stop if OOS exceeds total data
            if oos_end >= total_bars:
                oos_end = total_bars - 1
            if oos_start > total_bars - 1 or is_end >= total_bars - 1:
                logger.debug("Stopping rolling fold generation at fold %d.", i + 1)
                break

            folds.append(RollingFold(
                fold_number    = i + 1,
                is_start_bar   = is_start,
                is_end_bar     = is_end,
                oos_start_bar  = oos_start,
                oos_end_bar    = oos_end,
                mode           = FoldMode.ROLLING,
            ))

        return folds

    # ------------------------------------------------------------------
    # Date range resolution
    # ------------------------------------------------------------------

    def get_date_range(
        self,
        fold: AnyFold,
        index: Optional[pd.Index] = None,
    ) -> Tuple[str, str, str, str]:
        """
        Return (is_start, is_end, oos_start, oos_end) as ISO-8601 strings.

        If *index* is provided (a DatetimeIndex), the bar positions are
        translated to actual timestamps.  Otherwise falls back to the
        pre-set date strings on the fold or returns empty strings.

        Parameters
        ----------
        fold  : AnchoredFold or RollingFold
        index : optional pd.DatetimeIndex (same length as underlying data)

        Returns
        -------
        Tuple of four ISO-8601 strings: (is_start, is_end, oos_start, oos_end)
        """
        if index is not None and len(index) > 0:
            def _to_iso(bar: int) -> str:
                bar = max(0, min(bar, len(index) - 1))
                ts = index[bar]
                if hasattr(ts, "isoformat"):
                    return ts.isoformat()
                return str(ts)

            is_start  = _to_iso(fold.is_start_bar)
            is_end    = _to_iso(fold.is_end_bar)
            oos_start = _to_iso(fold.oos_start_bar)
            oos_end   = _to_iso(fold.oos_end_bar)

            # Cache on fold
            fold.is_start_date  = is_start
            fold.is_end_date    = is_end
            fold.oos_start_date = oos_start
            fold.oos_end_date   = oos_end

            return is_start, is_end, oos_start, oos_end

        # Fall back to pre-set strings
        return (
            fold.is_start_date,
            fold.is_end_date,
            fold.oos_start_date,
            fold.oos_end_date,
        )

    # ------------------------------------------------------------------
    # DataFrame splitting
    # ------------------------------------------------------------------

    def split_data(
        self,
        df: pd.DataFrame,
        fold: AnyFold,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split a DataFrame into IS and OOS portions based on fold bar indices.

        Validates no lookahead bias: the OOS portion must begin strictly
        after the IS portion ends.

        Parameters
        ----------
        df   : full DataFrame (rows are sequential bars)
        fold : AnchoredFold or RollingFold

        Returns
        -------
        (is_df, oos_df) — both are copies, never views.

        Raises
        ------
        ValueError  if lookahead bias is detected
        IndexError  if fold bar indices exceed DataFrame bounds
        """
        n = len(df)
        if n == 0:
            raise ValueError("Cannot split an empty DataFrame.")

        # Clamp to valid range
        is_start  = max(0, fold.is_start_bar)
        is_end    = min(fold.is_end_bar, n - 1)
        oos_start = max(0, fold.oos_start_bar)
        oos_end   = min(fold.oos_end_bar, n - 1)

        # Strict lookahead check
        if oos_start <= is_end:
            raise ValueError(
                f"Fold {fold.fold_number}: OOS start ({oos_start}) overlaps IS end "
                f"({is_end}).  Lookahead bias detected!"
            )

        is_df  = df.iloc[is_start  : is_end  + 1].copy()
        oos_df = df.iloc[oos_start : oos_end + 1].copy()

        logger.debug(
            "Fold %d split: IS [%d..%d]=%d bars, OOS [%d..%d]=%d bars.",
            fold.fold_number,
            is_start, is_end, len(is_df),
            oos_start, oos_end, len(oos_df),
        )
        return is_df, oos_df

    # ------------------------------------------------------------------
    # Month-based convenience
    # ------------------------------------------------------------------

    @staticmethod
    def bars_from_months(
        months: int,
        bars_per_day: float = 24.0,   # hourly bars for crypto
        trading_days_per_month: float = 30.4,
    ) -> int:
        """
        Convert a number of months to an approximate bar count.

        Parameters
        ----------
        months                  : number of months
        bars_per_day            : bars per trading day (default 24 for hourly crypto)
        trading_days_per_month  : assumed trading days per month

        Returns
        -------
        int — estimated total bars
        """
        return max(1, int(months * trading_days_per_month * bars_per_day))

    @classmethod
    def create_monthly_folds(
        cls,
        df: pd.DataFrame,
        n_folds: int = 8,
        in_sample_months: int = 12,
        out_sample_months: int = 3,
        mode: FoldMode = FoldMode.ANCHORED,
        gap_bars: int = 0,
    ) -> Tuple["FoldManager", List[AnyFold]]:
        """
        Convenience factory: create folds from a DataFrame using month counts.

        Detects bar frequency from the DataFrame's DatetimeIndex (if present),
        otherwise assumes hourly bars.

        Parameters
        ----------
        df                : full OHLCV DataFrame
        n_folds           : number of folds
        in_sample_months  : IS window length in months
        out_sample_months : OOS window length in months
        mode              : ANCHORED or ROLLING
        gap_bars          : gap bars between IS end and OOS start

        Returns
        -------
        (FoldManager, list of folds)
        """
        fm = cls(gap_bars=gap_bars)

        # Detect bars-per-day from index if possible
        bars_per_day = 24.0  # default: hourly crypto
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
            median_diff = pd.Series(df.index).diff().median()
            if pd.notna(median_diff):
                hours = median_diff.total_seconds() / 3600.0
                if hours > 0:
                    bars_per_day = 24.0 / hours

        is_bars  = fm.bars_from_months(in_sample_months, bars_per_day)
        oos_bars = fm.bars_from_months(out_sample_months, bars_per_day)

        total_bars = len(df)
        in_sample_ratio = is_bars / (is_bars + oos_bars)

        folds = fm.create_folds(
            total_bars      = total_bars,
            n_folds         = n_folds,
            in_sample_ratio = in_sample_ratio,
            mode            = mode,
        )

        # Populate date strings if DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            for fold in folds:
                fm.get_date_range(fold, df.index)

        return fm, folds

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self, folds: List[AnyFold]) -> str:
        """
        Return a human-readable summary of fold layout.

        Parameters
        ----------
        folds : list returned by create_folds

        Returns
        -------
        str
        """
        if not folds:
            return "No folds."

        lines = [
            f"Walk-Forward Folds  ({folds[0].mode.value}, gap={self.gap_bars})",
            f"{'Fold':>5}  {'IS start':>10}  {'IS end':>10}  "
            f"{'OOS start':>10}  {'OOS end':>10}  {'IS bars':>8}  {'OOS bars':>9}",
            "─" * 75,
        ]
        for f in folds:
            is_s  = f.is_start_date  or str(f.is_start_bar)
            is_e  = f.is_end_date    or str(f.is_end_bar)
            oos_s = f.oos_start_date or str(f.oos_start_bar)
            oos_e = f.oos_end_date   or str(f.oos_end_bar)
            lines.append(
                f"{f.fold_number:>5}  {is_s[:10]:>10}  {is_e[:10]:>10}  "
                f"{oos_s[:10]:>10}  {oos_e[:10]:>10}  "
                f"{f.is_bars:>8,}  {f.oos_bars:>9,}"
            )
        return "\n".join(lines)

    def coverage_stats(self, folds: List[AnyFold], total_bars: int) -> Dict[str, Any]:
        """
        Compute coverage statistics for a fold set.

        Parameters
        ----------
        folds       : list of folds
        total_bars  : total bars in the underlying dataset

        Returns
        -------
        dict with keys: n_folds, total_is_bars, total_oos_bars,
                        oos_coverage_pct, overlap_fraction
        """
        if not folds:
            return {}

        total_is  = sum(f.is_bars for f in folds)
        total_oos = sum(f.oos_bars for f in folds)

        # Unique OOS bars (no double-counting for rolling folds)
        oos_bars_set: set = set()
        for f in folds:
            oos_bars_set.update(range(f.oos_start_bar, f.oos_end_bar + 1))
        unique_oos = len(oos_bars_set)

        return {
            "n_folds":           len(folds),
            "total_is_bars":     total_is,
            "total_oos_bars":    total_oos,
            "unique_oos_bars":   unique_oos,
            "oos_coverage_pct":  round(unique_oos / max(total_bars, 1) * 100, 2),
            "overlap_fraction":  round((total_oos - unique_oos) / max(total_oos, 1), 4),
        }

    # ------------------------------------------------------------------
    # Date-range based creation (when you have a DatetimeIndex)
    # ------------------------------------------------------------------

    @classmethod
    def from_date_range(
        cls,
        df: pd.DataFrame,
        n_folds: int = 8,
        in_sample_months: int = 12,
        out_sample_months: int = 3,
        mode: FoldMode = FoldMode.ANCHORED,
        gap_bars: int = 0,
    ) -> Tuple["FoldManager", List[AnyFold]]:
        """
        Alias for create_monthly_folds — preferred entry point when a
        DataFrame with a DatetimeIndex is available.

        Returns
        -------
        (FoldManager, list of folds)
        """
        return cls.create_monthly_folds(
            df                = df,
            n_folds           = n_folds,
            in_sample_months  = in_sample_months,
            out_sample_months = out_sample_months,
            mode              = mode,
            gap_bars          = gap_bars,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def validate_no_overlap(folds: List[AnyFold]) -> bool:
        """
        Check that OOS windows do not overlap across folds (relevant for
        rolling mode only).

        Returns True if there is no OOS-OOS overlap.
        """
        seen: set = set()
        for fold in folds:
            oos_set = set(range(fold.oos_start_bar, fold.oos_end_bar + 1))
            if oos_set & seen:
                logger.warning(
                    "OOS overlap detected at fold %d.", fold.fold_number
                )
                return False
            seen.update(oos_set)
        return True

    @staticmethod
    def min_data_required(
        n_folds: int,
        in_sample_months: int,
        out_sample_months: int,
        bars_per_day: float = 24.0,
    ) -> int:
        """
        Compute the minimum number of bars required to build a valid fold set.

        Parameters
        ----------
        n_folds           : number of folds
        in_sample_months  : IS window in months
        out_sample_months : OOS window in months
        bars_per_day      : bar frequency

        Returns
        -------
        int — minimum bar count
        """
        is_bars  = FoldManager.bars_from_months(in_sample_months, bars_per_day)
        oos_bars = FoldManager.bars_from_months(out_sample_months, bars_per_day)
        # Need IS + n_folds * OOS for anchored WFA
        return is_bars + n_folds * oos_bars
