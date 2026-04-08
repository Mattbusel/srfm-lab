"""
Combinatorial Purged Cross-Validation (CPCV) Walk-Forward (T2-7)
Lopez de Prado method for unbiased strategy validation.

Standard walk-forward has leakage issues. CPCV fixes this via:
  - Purge: remove N bars before/after each test window (prevent label overlap)
  - Embargo: add M bars after training window before test (prevent microstructure leakage)
  - Combinatorial: all valid (train, test) splits are evaluated

Usage:
    validator = CPCVValidator(n_folds=6, purge_bars=16, embargo_bars=4)
    results = validator.validate(
        data=price_df,          # (n_bars,) price series
        strategy_fn=my_strategy  # fn(train_data) → BacktestResult
    )
"""
import math
import logging
import itertools
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

log = logging.getLogger(__name__)

@dataclass
class CVFold:
    fold_id: int
    train_indices: list[int]
    test_indices: list[int]
    purge_indices: list[int]
    embargo_indices: list[int]

@dataclass
class CPCVConfig:
    n_folds: int = 6
    purge_bars: int = 16         # bars purged before/after test window
    embargo_bars: int = 4        # bars embargoed after training window
    n_test_folds: int = 2        # test paths per split
    min_train_pct: float = 0.60  # minimum % of data in training set

@dataclass
class FoldResult:
    fold_id: int
    is_sharpe: float
    oos_sharpe: float
    is_return: float
    oos_return: float
    n_trades_is: int
    n_trades_oos: int
    degradation: float  # oos_sharpe - is_sharpe (negative = degradation)

@dataclass
class CPCVResult:
    fold_results: list[FoldResult]
    mean_oos_sharpe: float
    std_oos_sharpe: float
    deflated_sharpe: float        # DSR-adjusted Sharpe
    mean_degradation: float       # avg IS-OOS degradation
    overfitting_detected: bool
    n_splits: int

class CPCVValidator:
    """
    Combinatorial Purged Cross-Validation implementation.

    Produces multiple OOS paths from the same data without leakage.
    Key outputs: Deflated Sharpe Ratio (accounts for multiple testing),
    and mean IS-OOS degradation (measures overfitting severity).
    """

    def __init__(self, cfg: CPCVConfig = None):
        self.cfg = cfg or CPCVConfig()

    def generate_folds(self, n_bars: int) -> list[CVFold]:
        """Generate all CPCV train/test splits."""
        k = self.cfg.n_folds
        n_test = self.cfg.n_test_folds

        # Divide data into k groups
        group_size = n_bars // k
        groups = list(range(k))

        folds = []
        fold_id = 0

        # All combinations of n_test groups as test sets
        for test_groups in itertools.combinations(groups, n_test):
            train_groups = [g for g in groups if g not in test_groups]

            # Compute indices for each group
            test_indices = []
            for g in test_groups:
                start = g * group_size
                end = (g + 1) * group_size if g < k - 1 else n_bars
                test_indices.extend(range(start, end))

            train_indices_raw = []
            for g in train_groups:
                start = g * group_size
                end = (g + 1) * group_size if g < k - 1 else n_bars
                train_indices_raw.extend(range(start, end))

            # Purge: remove N bars around test window boundaries
            test_set = set(test_indices)
            purge_indices = set()
            for idx in test_indices:
                for d in range(-self.cfg.purge_bars, self.cfg.purge_bars + 1):
                    purge_indices.add(idx + d)

            # Embargo: remove M bars after each training→test transition
            # Find training bars just before test windows
            embargo_indices = set()
            for idx in test_indices:
                # Add embargo bars just before test window
                for d in range(self.cfg.embargo_bars):
                    embargo_indices.add(idx - 1 - d)

            # Clean train indices: remove purged and embargoed
            train_indices = [
                i for i in train_indices_raw
                if i not in purge_indices and i not in embargo_indices and 0 <= i < n_bars
            ]

            # Check minimum train size
            if len(train_indices) < int(n_bars * self.cfg.min_train_pct):
                continue

            folds.append(CVFold(
                fold_id=fold_id,
                train_indices=sorted(train_indices),
                test_indices=sorted([i for i in test_indices if 0 <= i < n_bars]),
                purge_indices=sorted(purge_indices),
                embargo_indices=sorted(embargo_indices),
            ))
            fold_id += 1

        log.info("CPCV: generated %d folds from %d bars (k=%d, n_test=%d, purge=%d, embargo=%d)",
                 len(folds), n_bars, k, n_test, self.cfg.purge_bars, self.cfg.embargo_bars)
        return folds

    def validate(
        self,
        n_bars: int,
        strategy_fn: Callable,  # fn(train_idx, test_idx) → (is_sharpe, oos_sharpe, is_ret, oos_ret, n_trades_is, n_trades_oos)
    ) -> CPCVResult:
        """
        Run CPCV validation.

        strategy_fn receives (train_indices, test_indices) and returns
        (is_sharpe, oos_sharpe, is_return, oos_return, n_trades_is, n_trades_oos).
        """
        folds = self.generate_folds(n_bars)
        if not folds:
            log.warning("CPCV: no valid folds generated")
            return CPCVResult([], 0.0, 0.0, 0.0, 0.0, True, 0)

        fold_results = []
        for fold in folds:
            try:
                result = strategy_fn(fold.train_indices, fold.test_indices)
                if result is None:
                    continue
                is_sh, oos_sh, is_ret, oos_ret, n_is, n_oos = result
                fold_results.append(FoldResult(
                    fold_id=fold.fold_id,
                    is_sharpe=float(is_sh),
                    oos_sharpe=float(oos_sh),
                    is_return=float(is_ret),
                    oos_return=float(oos_ret),
                    n_trades_is=int(n_is),
                    n_trades_oos=int(n_oos),
                    degradation=float(oos_sh) - float(is_sh),
                ))
            except Exception as e:
                log.warning("CPCV: fold %d failed: %s", fold.fold_id, e)

        if not fold_results:
            return CPCVResult([], 0.0, 0.0, 0.0, 0.0, True, 0)

        oos_sharpes = [r.oos_sharpe for r in fold_results]
        n = len(oos_sharpes)
        mean_oos = sum(oos_sharpes) / n
        std_oos = (sum((s - mean_oos)**2 for s in oos_sharpes) / n) ** 0.5

        degradations = [r.degradation for r in fold_results]
        mean_degrad = sum(degradations) / len(degradations)

        # Deflated Sharpe Ratio (DSR): adjusts for multiple testing
        # DSR = SR * (1 - gamma(0.5) * sqrt(variance) / sqrt(n))
        # Simplified version:
        deflated_sharpe = mean_oos * (1.0 - (std_oos / (abs(mean_oos) + 1e-6)) * 0.5 / math.sqrt(n))

        overfitting = mean_degrad < -0.3 or deflated_sharpe < 0.0

        log.info(
            "CPCV: %d folds, mean_OOS_Sharpe=%.3f ± %.3f, DSR=%.3f, mean_degradation=%.3f, overfitting=%s",
            n, mean_oos, std_oos, deflated_sharpe, mean_degrad, overfitting
        )

        return CPCVResult(
            fold_results=fold_results,
            mean_oos_sharpe=mean_oos,
            std_oos_sharpe=std_oos,
            deflated_sharpe=deflated_sharpe,
            mean_degradation=mean_degrad,
            overfitting_detected=overfitting,
            n_splits=n,
        )
