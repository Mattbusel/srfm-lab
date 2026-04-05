"""
training/cross_validator.py
============================
Purged K-fold and Combinatorial Purged Cross-Validation (CPCV)
for financial time-series.

Financial rationale
-------------------
The bias introduced by using standard k-fold on a time series is
subtle but large in magnitude.  Suppose a daily series has overlapping
5-bar labels (e.g. 5-day forward returns).  A validation observation at
bar 100 has a label that uses bars 101-105.  If bar 103 is in the
training set, the model has indirect access to validation label
information through its own training loss.  This inflates in-sample IC
estimates by ~30 % in typical backtests.

Purging (Marcos Lopez de Prado, 2018):
    Remove any training observation whose label forward window overlaps
    the validation window.

Embargo:
    Additionally remove ``embargo_bars`` observations immediately before
    and after the validation window to account for autocorrelated features
    (e.g. a 20-bar EMA at bar t still "knows about" bars t-19…t-1, some
    of which may be in the validation set).

CPCV (Combinatorial Purged CV):
    Standard purged k-fold only produces k test paths.  CPCV generates
    C(k, n_test) paths by treating all combinations of n_test folds as
    the test set.  This produces a better estimate of out-of-sample
    distribution because we see many more independent test paths.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterator, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, spearmanr


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class CVResult(NamedTuple):
    fold:       int
    n_train:    int
    n_val:      int
    ic:         float
    icir:       float
    t_stat:     float
    p_value:    float
    predictions: np.ndarray
    actuals:     np.ndarray


# ---------------------------------------------------------------------------
# PurgedCrossValidator
# ---------------------------------------------------------------------------

class PurgedCrossValidator:
    """Purged K-fold and CPCV for ML signal evaluation.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    horizon : int
        Forward label horizon (used for purging).
    embargo_bars : int
        Bars to remove around fold boundaries.
    """

    def __init__(
        self,
        n_splits:     int = 5,
        horizon:      int = 1,
        embargo_bars: int = 5,
    ) -> None:
        self.n_splits     = n_splits
        self.horizon      = horizon
        self.embargo_bars = embargo_bars

    # ------------------------------------------------------------------
    # Split generators
    # ------------------------------------------------------------------

    def split(
        self, df: pd.DataFrame
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_positions, val_positions) with purging + embargo."""
        N         = len(df)
        indices   = np.arange(N)
        fold_size = N // self.n_splits

        for fold in range(self.n_splits):
            val_s = fold * fold_size
            val_e = val_s + fold_size if fold < self.n_splits - 1 else N

            val_idx = indices[val_s:val_e]

            # Purge: training obs whose label window overlaps val period
            # Label for obs i covers bars i+1 … i+horizon
            purge_mask = (indices + self.horizon) >= val_s
            embargo_s  = max(0, val_s - self.embargo_bars)
            embargo_e  = min(N, val_e + self.embargo_bars)

            train_mask = (
                ~purge_mask
                | (indices < embargo_s)
                | (indices >= embargo_e)
            )
            # Clean version: exclude [val_s - horizon - embargo, val_e + embargo]
            exclude_s  = max(0, val_s - self.horizon - self.embargo_bars)
            exclude_e  = min(N, val_e + self.embargo_bars)
            train_mask = (indices < exclude_s) | (indices >= exclude_e)

            train_idx = indices[train_mask]
            yield train_idx, val_idx

    def cpcv_split(
        self,
        df:            pd.DataFrame,
        n_test_splits: int = 2,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Combinatorial purged cross-validation splits.

        Yields C(n_splits, n_test_splits) (train, test) pairs.
        """
        N         = len(df)
        indices   = np.arange(N)
        fold_size = N // self.n_splits
        fold_ranges = [
            (k * fold_size, min((k + 1) * fold_size, N))
            for k in range(self.n_splits)
        ]

        for test_combo in combinations(range(self.n_splits), n_test_splits):
            test_ranges  = [fold_ranges[k] for k in test_combo]
            test_set     = set()
            for s, e in test_ranges:
                test_set.update(range(s, e))
            test_idx = np.array(sorted(test_set))

            train_idx_list = []
            for i in indices:
                if i in test_set:
                    continue
                label_end = i + self.horizon
                overlaps  = any(s <= label_end < e for s, e in test_ranges)
                near_bnd  = any(
                    abs(i - s) <= self.embargo_bars or abs(i - e) <= self.embargo_bars
                    for s, e in test_ranges
                )
                if not overlaps and not near_bnd:
                    train_idx_list.append(i)

            if train_idx_list:
                yield np.array(train_idx_list), test_idx

    # ------------------------------------------------------------------
    # Cross-validation runner
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        df:            pd.DataFrame,
        model_class,
        model_kwargs:  dict,
        predict_fn     = None,
        cpcv:          bool = False,
        n_test_splits: int  = 2,
    ) -> List[CVResult]:
        """Run cross-validation and collect per-fold metrics.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame with ``target`` column.
        model_class : type
            Model constructor.
        model_kwargs : dict
            Constructor kwargs.
        predict_fn : callable, optional
            Custom function (model, df) → array.  Defaults to batch predict.
        cpcv : bool
            If True, use CPCV splits instead of standard purged k-fold.
        n_test_splits : int
            CPCV parameter.

        Returns
        -------
        List[CVResult]
        """
        split_iter = (
            self.cpcv_split(df, n_test_splits)
            if cpcv
            else self.split(df)
        )
        results = []
        for fold_i, (train_idx, val_idx) in enumerate(split_iter):
            train_df = df.iloc[train_idx].copy()
            val_df   = df.iloc[val_idx].copy()

            if len(train_df) < 30 or "target" not in df.columns:
                continue

            model = model_class(**model_kwargs)
            try:
                model.fit(train_df)
            except Exception:
                continue

            # Predictions
            if predict_fn is not None:
                preds = predict_fn(model, val_df)
            else:
                preds = self._batch_predict(model, val_df)

            actuals = val_df["target"].values
            n = min(len(preds), len(actuals))
            preds   = preds[:n]
            actuals = actuals[:n]

            if n < 5:
                continue

            ic, _   = spearmanr(preds, actuals)
            ic      = float(ic) if not np.isnan(ic) else 0.0

            # Rolling IC std for ICIR
            window = min(20, n // 2)
            ics = []
            for i in range(window, n):
                p_w = preds[i - window:i]
                r_w = actuals[i - window:i]
                if np.std(p_w) > 1e-9 and np.std(r_w) > 1e-9:
                    ic_w, _ = spearmanr(p_w, r_w)
                    if not np.isnan(ic_w):
                        ics.append(ic_w)

            ic_arr  = np.array(ics) if ics else np.array([ic])
            icir    = float(ic_arr.mean() / (ic_arr.std() + 1e-9))
            t_stat, p_val = ttest_1samp(ic_arr, 0.0)

            results.append(CVResult(
                fold        = fold_i,
                n_train     = len(train_df),
                n_val       = n,
                ic          = ic,
                icir        = icir,
                t_stat      = float(t_stat),
                p_value     = float(p_val),
                predictions = preds,
                actuals     = actuals,
            ))

        return results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summarise(self, results: List[CVResult]) -> dict:
        """Aggregate cross-validation results into a summary dict."""
        if not results:
            return {}
        ics   = np.array([r.ic for r in results])
        icirs = np.array([r.icir for r in results])
        t_agg, p_agg = ttest_1samp(ics, 0.0)
        return {
            "mean_ic":    float(ics.mean()),
            "std_ic":     float(ics.std()),
            "min_ic":     float(ics.min()),
            "max_ic":     float(ics.max()),
            "mean_icir":  float(icirs.mean()),
            "t_stat_ic":  float(t_agg),
            "p_value_ic": float(p_agg),
            "n_folds":    len(results),
            "pct_positive_ic": float((ics > 0).mean()),
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _batch_predict(model, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for i in range(len(df)):
            try:
                p = model.predict(df.iloc[:i + 1])
            except Exception:
                p = 0.0
            preds.append(float(p))
        return np.array(preds)
