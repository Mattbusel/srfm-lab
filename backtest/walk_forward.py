"""
walk_forward.py -- Walk-forward optimization and time-series cross-validation.

Implements sliding-window train/test splits, purged K-fold with embargo,
parameter grid search, and the deflated Sharpe ratio overfit detector.
"""

from __future__ import annotations

import itertools
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Walk-Forward Splits
# ---------------------------------------------------------------------------

@dataclass
class WFOSplit:
    """A single train/test split for walk-forward optimization."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_start: pd.Timestamp  # gap between train and test

    @property
    def train_len(self) -> pd.Timedelta:
        return self.train_end - self.train_start

    @property
    def test_len(self) -> pd.Timedelta:
        return self.test_end - self.test_start

    def __repr__(self) -> str:
        return (
            f"WFOSplit(fold={self.fold_id}, "
            f"train=[{self.train_start.date()}, {self.train_end.date()}], "
            f"test=[{self.test_start.date()}, {self.test_end.date()}])"
        )


class WalkForwardOptimizer:
    """
    Sliding-window walk-forward optimizer for the LARSA strategy.

    Splits a time series into overlapping train/test windows.
    Optionally uses anchored (expanding) rather than rolling train windows.

    Parameters
    ----------
    train_bars : int
        Number of bars in each training window.
    test_bars : int
        Number of bars in each test (out-of-sample) window.
    step_bars : int
        Number of bars to slide the window forward on each step.
        Set equal to test_bars for non-overlapping test sets.
    embargo_bars : int
        Bars of gap between train end and test start (prevent leakage).
    anchored : bool
        If True, the training window expands from the start rather than sliding.
    """

    def __init__(
        self,
        train_bars: int = 2000,
        test_bars: int = 500,
        step_bars: Optional[int] = None,
        embargo_bars: int = 8,
        anchored: bool = False,
    ):
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars or test_bars
        self.embargo_bars = embargo_bars
        self.anchored = anchored

    def split(self, index: pd.DatetimeIndex) -> List[WFOSplit]:
        """
        Generate all WFOSplits for the given index.
        Each split is a named tuple of (train_start, train_end, test_start, test_end).
        """
        n = len(index)
        min_required = self.train_bars + self.embargo_bars + self.test_bars
        if n < min_required:
            raise ValueError(
                f"Index too short: {n} bars, need at least {min_required}"
            )

        splits = []
        fold_id = 0
        test_start_idx = self.train_bars + self.embargo_bars

        while test_start_idx + self.test_bars <= n:
            if self.anchored:
                train_start_idx = 0
            else:
                train_start_idx = test_start_idx - self.embargo_bars - self.train_bars

            train_end_idx = test_start_idx - self.embargo_bars - 1
            embargo_start_idx = train_end_idx + 1
            test_end_idx = min(test_start_idx + self.test_bars - 1, n - 1)

            split = WFOSplit(
                fold_id=fold_id,
                train_start=index[train_start_idx],
                train_end=index[train_end_idx],
                test_start=index[test_start_idx],
                test_end=index[test_end_idx],
                embargo_start=index[embargo_start_idx],
            )
            splits.append(split)
            fold_id += 1
            test_start_idx += self.step_bars

        logger.info("Generated %d WFO splits", len(splits))
        return splits

    def get_data_for_split(
        self, data: pd.DataFrame, split: WFOSplit
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (train_df, test_df) for a given split."""
        train = data.loc[split.train_start : split.train_end]
        test = data.loc[split.test_start : split.test_end]
        return train, test

    def run(
        self,
        data: pd.DataFrame,
        objective_fn: Callable[[pd.DataFrame, Dict], float],
        param_grid: "ParameterGrid",
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """
        Full WFO run.
        1. For each fold, find the best parameters on the train set.
        2. Evaluate on the out-of-sample test set.
        Returns a DataFrame of per-fold results.
        """
        splits = self.split(data.index)
        results = []

        for split in splits:
            train, test = self.get_data_for_split(data, split)

            # Find best params on train
            best_score = -np.inf
            best_params = None
            for params in param_grid:
                try:
                    score = objective_fn(train, params)
                    if score > best_score:
                        best_score = score
                        best_params = params
                except Exception as exc:
                    logger.debug("Param eval failed: %s | %s", params, exc)

            # Evaluate on test
            oos_score = None
            if best_params is not None:
                try:
                    oos_score = objective_fn(test, best_params)
                except Exception as exc:
                    logger.warning("OOS eval failed for fold %d: %s", split.fold_id, exc)

            results.append(
                {
                    "fold_id": split.fold_id,
                    "train_start": split.train_start,
                    "train_end": split.train_end,
                    "test_start": split.test_start,
                    "test_end": split.test_end,
                    "best_params": best_params,
                    "is_sharpe": best_score,
                    "oos_sharpe": oos_score,
                }
            )

        return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Purged K-Fold with Embargo
# ---------------------------------------------------------------------------

@dataclass
class PurgedKFoldSplit:
    """A single fold from PurgedKFold cross-validation."""
    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    purged_indices: np.ndarray  # indices removed from train due to proximity to test


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation for financial time series.

    Standard K-Fold has lookahead bias because test observations in
    the future may be correlated with training observations just before them.

    Purging removes training samples that overlap (in time) with the test set.
    Embargo adds a gap after the test set before training resumes.

    Reference: Lopez de Prado (2018), Chapter 7.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,  # embargo = 1% of total samples
    ):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self, X: pd.DataFrame
    ) -> Generator[PurgedKFoldSplit, None, None]:
        """
        Generate purged train/test splits.

        Yields PurgedKFoldSplit objects for each fold.
        """
        n = len(X)
        embargo_size = max(1, int(n * self.embargo_pct))
        fold_size = n // self.n_splits

        for fold_id in range(self.n_splits):
            # Test set: fold_id * fold_size to (fold_id+1) * fold_size
            test_start = fold_id * fold_size
            test_end = (fold_id + 1) * fold_size if fold_id < self.n_splits - 1 else n
            test_idx = np.arange(test_start, test_end)

            # Training set: all except test
            all_idx = np.arange(n)
            train_idx = np.setdiff1d(all_idx, test_idx)

            # Purge: remove training samples within embargo of test set boundaries
            embargo_start = test_start - embargo_size
            embargo_end = test_end + embargo_size

            purged_mask = (train_idx >= embargo_start) & (train_idx < embargo_end)
            purged_idx = train_idx[purged_mask]
            train_idx_clean = train_idx[~purged_mask]

            yield PurgedKFoldSplit(
                fold_id=fold_id,
                train_indices=train_idx_clean,
                test_indices=test_idx,
                purged_indices=purged_idx,
            )

    def cross_val_score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scorer: Callable[[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], float],
    ) -> np.ndarray:
        """
        Run cross-validation and return array of scores.
        scorer(X_train, y_train, X_test, y_test) -> float
        """
        scores = []
        for split in self.split(X):
            X_train = X.iloc[split.train_indices]
            y_train = y.iloc[split.train_indices] if y is not None else None
            X_test = X.iloc[split.test_indices]
            y_test = y.iloc[split.test_indices] if y is not None else None
            try:
                score = scorer(X_train, y_train, X_test, y_test)
                scores.append(score)
            except Exception as exc:
                logger.warning("Fold %d failed: %s", split.fold_id, exc)
                scores.append(np.nan)
        return np.array(scores)

    def validate_no_lookahead(self, splits: List[PurgedKFoldSplit]) -> bool:
        """
        Verify that no training sample occurs after the start of the test set
        without embargo protection. Returns True if no lookahead detected.
        """
        for split in splits:
            test_start = split.test_indices.min() if len(split.test_indices) > 0 else np.inf
            # Check if any unpurged train indices are >= test_start
            if len(split.train_indices) > 0:
                max_train = split.train_indices.max()
                # Allow overlap up to test_start + embargo (purged region handles this)
                purged_up_to = split.purged_indices.min() if len(split.purged_indices) > 0 else test_start
                if max_train >= test_start and max_train >= purged_up_to:
                    logger.error(
                        "Lookahead detected in fold %d: max_train=%d, test_start=%d",
                        split.fold_id, max_train, test_start,
                    )
                    return False
        return True


# ---------------------------------------------------------------------------
# Parameter Grid
# ---------------------------------------------------------------------------

class ParameterGrid:
    """
    Exhaustive and random search over LARSA parameter space.

    Supports:
      - Grid search: all combinations of provided parameter ranges
      - Random search: random samples from parameter distributions
      - Quasi-random: Sobol sequence for more uniform coverage
    """

    # Default LARSA parameter space
    DEFAULT_LARSA_SPACE: Dict[str, Any] = {
        "bh_decay": [0.90, 0.92, 0.94, 0.96, 0.98],
        "bh_mass_threshold": [0.001, 0.002, 0.0025, 0.003, 0.005],
        "cf_fast_period": [5, 8, 13, 21],
        "cf_slow_period": [13, 21, 34, 55],
        "hurst_min_window": [24, 32, 48, 64],
        "garch_alpha": [0.05, 0.08, 0.10, 0.12, 0.15],
        "garch_beta": [0.80, 0.83, 0.85, 0.87, 0.90],
        "kelly_fraction": [0.15, 0.20, 0.25, 0.30],
        "target_annual_vol": [0.10, 0.12, 0.15, 0.20],
        "ramp_bars": [2, 3, 4, 6],
        "min_hold_bars": [2, 4, 8, 12],
    }

    def __init__(
        self,
        param_space: Optional[Dict[str, Any]] = None,
        mode: str = "grid",
        n_random: int = 100,
        seed: int = 42,
    ):
        self.param_space = param_space or self.DEFAULT_LARSA_SPACE
        self.mode = mode
        self.n_random = n_random
        self.rng = np.random.default_rng(seed)
        self._grid: Optional[List[Dict]] = None

    def _build_grid(self) -> List[Dict]:
        """Build the full Cartesian product grid."""
        keys = list(self.param_space.keys())
        values = list(self.param_space.values())
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]

    def _random_sample(self) -> List[Dict]:
        """Sample random combinations."""
        keys = list(self.param_space.keys())
        values = list(self.param_space.values())
        samples = []
        for _ in range(self.n_random):
            sample = {k: self.rng.choice(v) for k, v in zip(keys, values)}
            samples.append(sample)
        return samples

    def __iter__(self) -> Iterator[Dict]:
        if self.mode == "grid":
            if self._grid is None:
                self._grid = self._build_grid()
            return iter(self._grid)
        elif self.mode == "random":
            return iter(self._random_sample())
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def __len__(self) -> int:
        if self.mode == "grid":
            n = 1
            for v in self.param_space.values():
                n *= len(v)
            return n
        else:
            return self.n_random

    def subset(self, n: int) -> "ParameterGrid":
        """Return a grid with only n random samples."""
        new_grid = ParameterGrid(self.param_space, mode="random", n_random=n, seed=int(self.rng.integers(1000)))
        return new_grid

    def validate_params(self, params: Dict) -> bool:
        """Basic LARSA parameter sanity checks."""
        errors = []
        cf_fast = params.get("cf_fast_period", 8)
        cf_slow = params.get("cf_slow_period", 21)
        if cf_fast >= cf_slow:
            errors.append(f"cf_fast ({cf_fast}) must be < cf_slow ({cf_slow})")

        alpha = params.get("garch_alpha", 0.10)
        beta = params.get("garch_beta", 0.85)
        if alpha + beta >= 1.0:
            errors.append(f"GARCH: alpha+beta={alpha+beta:.3f} >= 1.0 (non-stationary)")

        if errors:
            logger.debug("Invalid params: %s", errors)
            return False
        return True

    def filter_valid(self) -> "ParameterGrid":
        """Return a new grid containing only valid parameter combinations."""
        valid = [p for p in self if self.validate_params(p)]
        new_grid = ParameterGrid.__new__(ParameterGrid)
        new_grid.param_space = self.param_space
        new_grid.mode = "grid"
        new_grid._grid = valid
        new_grid.n_random = self.n_random
        new_grid.rng = self.rng
        return new_grid


# ---------------------------------------------------------------------------
# Overfit Detector (Deflated Sharpe Ratio test)
# ---------------------------------------------------------------------------

class OverfitDetector:
    """
    Detects parameter overfitting using the Deflated Sharpe Ratio (DSR) test.

    Reference: Bailey, D.H. & Lopez de Prado, M. (2014).
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting
    and Non-Normality."

    The key idea: if we test N parameter combinations and report the best one,
    the probability of a spurious positive Sharpe increases with N.
    DSR adjusts for this multiple-comparison bias.
    """

    def __init__(self, significance: float = 0.95):
        self.significance = significance

    def deflated_sharpe(
        self,
        sharpe_observed: float,
        returns: np.ndarray,
        n_trials: int = 1,
    ) -> float:
        """
        Compute DSR = Phi(z), the probability the strategy is genuinely profitable.

        Parameters
        ----------
        sharpe_observed : float
            The best observed (in-sample) Sharpe ratio.
        returns : array
            The return series corresponding to sharpe_observed.
        n_trials : int
            Number of strategy configurations tested (for multiple-testing correction).
        """
        n = len(returns)
        if n < 5:
            return 0.0

        skew = float(scipy_stats.skew(returns))
        kurt = float(scipy_stats.kurtosis(returns))  # excess

        # Expected max Sharpe from n_trials ~ E[max] approx sqrt(2 * log(n_trials))
        if n_trials > 1:
            expected_max_sr = _expected_max_sharpe(n_trials, n)
        else:
            expected_max_sr = 0.0

        # Variance of Sharpe estimator
        variance = (1 - skew * sharpe_observed + (kurt / 4.0) * sharpe_observed**2) / (n - 1)
        se = np.sqrt(max(variance, 1e-12))

        z_stat = (sharpe_observed - expected_max_sr) / se
        dsr = float(scipy_stats.norm.cdf(z_stat))
        return dsr

    def is_overfit(
        self,
        is_sharpe: float,
        oos_sharpe: float,
        n_trials: int = 1,
        returns: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Heuristic overfit detection:
        1. OOS Sharpe < 0 when IS Sharpe > 0 (strong signal of overfit)
        2. OOS/IS Sharpe ratio < 0.5 (OOS degrades by more than 50%)
        3. DSR < significance threshold
        """
        if is_sharpe <= 0:
            return False

        if oos_sharpe <= 0:
            return True

        degradation = oos_sharpe / is_sharpe
        if degradation < 0.5:
            return True

        if returns is not None:
            dsr = self.deflated_sharpe(is_sharpe, returns, n_trials)
            if dsr < self.significance:
                return True

        return False

    def overfit_probability(
        self,
        wfo_results: pd.DataFrame,
    ) -> float:
        """
        Given a DataFrame of WFO results with columns [is_sharpe, oos_sharpe],
        estimate the overall probability of overfitting.
        """
        if wfo_results.empty:
            return 1.0
        col_is = "is_sharpe"
        col_oos = "oos_sharpe"
        if col_is not in wfo_results.columns or col_oos not in wfo_results.columns:
            return 1.0

        valid = wfo_results[[col_is, col_oos]].dropna()
        if len(valid) == 0:
            return 1.0

        # Probability of overfitting (PBO) via distribution of IS vs OOS
        is_arr = valid[col_is].values
        oos_arr = valid[col_oos].values

        # Simple PBO: fraction of folds where OOS < IS by more than 50%
        ratios = oos_arr / (is_arr + 1e-12)
        pbo = float((ratios < 0.5).mean())
        return pbo


def _expected_max_sharpe(n_trials: int, n_obs: int) -> float:
    """
    Expected maximum Sharpe ratio from n_trials independent strategy tests,
    each estimated with n_obs observations.
    Approximation: E[max] ~ Phi_inv(1 - 1/n_trials) / sqrt(n_obs)
    """
    from scipy.special import ndtri
    if n_trials <= 1 or n_obs <= 1:
        return 0.0
    z = float(ndtri(1 - 1.0 / n_trials))
    return z / np.sqrt(n_obs)
