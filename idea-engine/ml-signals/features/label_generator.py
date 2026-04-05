"""
features/label_generator.py
============================
Target label generation for supervised ML signal training.

Financial rationale
-------------------
The choice of target variable has an enormous impact on what a model
learns and how it behaves live:

forward_return(n)
    The raw n-bar return.  Simple but sensitive to large outliers.
    Appropriate for regression models (LSTM magnitude head).

direction(n)
    Sign of forward_return.  Converts regression to binary classification.
    Information-theoretically weaker but much more stable to train on.
    Appropriate for XGBoost direction head.

sharpe_label(n, vol_window)
    forward_return / rolling_vol.  This is the "return per unit of risk"
    view.  A model trained on this target learns to favour moves where
    the expected payout is high *relative to* the current noise level.
    This is especially useful in crypto where volatility regimes shift
    dramatically.

triple_barrier(n, upper_pct, lower_pct)
    Lopez de Prado's triple-barrier method.  For each bar, look forward
    up to n bars and record which barrier is hit first:
        +1 → price rises by upper_pct before falling by lower_pct
        -1 → price falls by lower_pct before rising by upper_pct
         0 → neither barrier hit within n bars (time-out)
    This is the most realistic label because it mirrors how an actual
    stop-loss / take-profit strategy behaves.

Purged K-Fold Cross-Validation
-------------------------------
Financial time-series data has serial correlation: consecutive labels
can share future return information (the label at bar t uses returns
t+1…t+n, overlapping with the label at bar t+1 which uses t+2…t+n+1).
Leaking validation labels into training causes optimistically biased
IC estimates.

Purging removes training samples whose forward return window overlaps
with the validation period.  An additional embargo of ``embargo_bars``
is removed after each fold boundary to prevent indirect leakage through
feature momentum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# LabelGenerator
# ---------------------------------------------------------------------------

class LabelGenerator:
    """Generate supervised learning targets from price data.

    Parameters
    ----------
    horizon : int
        Default forward horizon (bars) for target computation.
    vol_window : int
        Rolling window for realised volatility used in sharpe_label.
    embargo_bars : int
        Number of bars to exclude after each fold boundary in CPCV.
    """

    def __init__(
        self,
        horizon: int = 1,
        vol_window: int = 20,
        embargo_bars: int = 5,
    ) -> None:
        self.horizon      = horizon
        self.vol_window   = vol_window
        self.embargo_bars = embargo_bars

    # ------------------------------------------------------------------
    # Target generators
    # ------------------------------------------------------------------

    def forward_return(self, df: pd.DataFrame, n: Optional[int] = None) -> pd.Series:
        """n-bar forward return: (close[t+n] - close[t]) / close[t].

        The returned Series has NaN for the last n rows (no future data).
        """
        n = n or self.horizon
        c = df["close"] if "close" in df.columns else df.iloc[:, 0]
        fwd = c.shift(-n) / c - 1.0
        return fwd.rename(f"fwd_ret_{n}")

    def direction(self, df: pd.DataFrame, n: Optional[int] = None) -> pd.Series:
        """Sign of forward_return: +1 (up) or -1 (down or flat)."""
        n = n or self.horizon
        fwd = self.forward_return(df, n)
        return np.sign(fwd).replace(0, -1).rename(f"direction_{n}")

    def sharpe_label(
        self, df: pd.DataFrame, n: Optional[int] = None,
        vol_window: Optional[int] = None
    ) -> pd.Series:
        """Forward return scaled by rolling realised vol.

        This is the ex-post Sharpe-like ratio: we reward the model when
        it correctly anticipates a high-return move that also occurred
        in a low-volatility regime.

        Parameters
        ----------
        n : int, optional
            Forward horizon.  Defaults to ``self.horizon``.
        vol_window : int, optional
            Rolling window for vol estimate.  Defaults to ``self.vol_window``.
        """
        n          = n or self.horizon
        vol_window = vol_window or self.vol_window
        c          = df["close"] if "close" in df.columns else df.iloc[:, 0]
        lr         = np.log(c / c.shift(1))
        rv         = lr.rolling(vol_window).std().replace(0, np.nan)
        fwd        = self.forward_return(df, n)
        return (fwd / rv).clip(-10, 10).rename(f"sharpe_label_{n}")

    def triple_barrier(
        self,
        df: pd.DataFrame,
        n: Optional[int] = None,
        upper_pct: float = 0.02,
        lower_pct: float = 0.02,
    ) -> pd.Series:
        """Lopez de Prado triple-barrier labelling.

        For each bar t, examine prices t+1 … t+n.  Return:
            +1  if  max(returns[t:t+k]) >= upper_pct first
            -1  if  min(returns[t:t+k]) <= -lower_pct first
             0  if  neither barrier hit within n bars

        Parameters
        ----------
        n : int
            Maximum holding period (bars).
        upper_pct : float
            Upper take-profit barrier as fraction of entry price.
        lower_pct : float
            Lower stop-loss barrier as fraction of entry price
            (expressed as positive, e.g. 0.02 = 2 % drop).
        """
        n  = n or self.horizon
        c  = (df["close"] if "close" in df.columns else df.iloc[:, 0]).values
        T  = len(c)
        labels = np.zeros(T, dtype=np.float64)

        for t in range(T - 1):
            entry  = c[t]
            result = 0
            for k in range(1, min(n + 1, T - t)):
                ret = (c[t + k] - entry) / (entry + 1e-9)
                if ret >= upper_pct:
                    result = 1
                    break
                if ret <= -lower_pct:
                    result = -1
                    break
            labels[t] = result

        # Last n bars: no valid label
        labels[T - n :] = np.nan
        return pd.Series(labels, index=df.index, name=f"triple_barrier_{n}")

    # ------------------------------------------------------------------
    # Purged K-Fold
    # ------------------------------------------------------------------

    def purged_kfold_splits(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        horizon: Optional[int] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, val_idx) for purged time-series K-fold.

        Training indices whose forward window overlaps the validation
        period are removed.  An embargo of ``self.embargo_bars`` is
        removed at each fold boundary to further prevent leakage.

        Parameters
        ----------
        df : pd.DataFrame
            Must have a monotonically increasing DatetimeIndex.
        n_splits : int
            Number of folds.
        horizon : int, optional
            Forward horizon used for purging.  Defaults to ``self.horizon``.

        Yields
        ------
        train_idx : np.ndarray  (integer positions)
        val_idx   : np.ndarray  (integer positions)
        """
        horizon = horizon or self.horizon
        N = len(df)
        indices = np.arange(N)
        fold_size = N // n_splits

        for fold in range(n_splits):
            val_start = fold * fold_size
            val_end   = val_start + fold_size if fold < n_splits - 1 else N

            val_idx = indices[val_start:val_end]

            # Purge: remove train samples whose label window overlaps val
            # Sample at position t has label horizon = [t+1 … t+horizon]
            # Overlap if t + horizon >= val_start
            purge_start = max(0, val_start - horizon)
            # Embargo: remove bars just before val_start and after val_end
            embargo_before = max(0, purge_start - self.embargo_bars)
            embargo_after  = min(N, val_end + self.embargo_bars)

            train_mask = (indices < embargo_before) | (indices >= embargo_after)
            train_idx  = indices[train_mask]

            yield train_idx, val_idx

    def cpcv_splits(
        self,
        df: pd.DataFrame,
        n_splits: int = 6,
        n_test_splits: int = 2,
        horizon: Optional[int] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Combinatorial Purged Cross-Validation (Lopez de Prado 2018).

        Generates C(n_splits, n_test_splits) train/test combinations,
        each with purging and embargo applied.  This produces a larger
        number of independent test paths than standard K-fold while
        preserving the temporal structure.

        Parameters
        ----------
        n_splits : int
            Total number of folds to split the data into.
        n_test_splits : int
            Number of folds combined to form each test set.
        horizon : int, optional
            Forward horizon for purging.
        """
        from itertools import combinations
        horizon = horizon or self.horizon
        N = len(df)
        indices = np.arange(N)
        fold_size = N // n_splits
        fold_ranges = [
            (k * fold_size, (k + 1) * fold_size if k < n_splits - 1 else N)
            for k in range(n_splits)
        ]

        for test_folds in combinations(range(n_splits), n_test_splits):
            test_ranges  = [fold_ranges[k] for k in test_folds]
            test_idx_set = set()
            for s, e in test_ranges:
                test_idx_set.update(range(s, e))
            test_idx = np.array(sorted(test_idx_set))

            # Purge: remove train rows whose horizon overlaps any test range
            train_idx_list = []
            for i in indices:
                if i in test_idx_set:
                    continue
                # Check if i's label window overlaps any test fold
                label_end = i + horizon
                overlap = any(s <= label_end and i < e for s, e in test_ranges)
                # Embargo: within embargo_bars of any test boundary
                near_boundary = any(
                    abs(i - s) < self.embargo_bars or abs(i - e) < self.embargo_bars
                    for s, e in test_ranges
                )
                if not overlap and not near_boundary:
                    train_idx_list.append(i)

            if len(train_idx_list) > 0:
                yield np.array(train_idx_list), test_idx
