"""
ml/training/cross_validator.py

Financial time series cross-validation for SRFM.
Implements purged k-fold CV, combinatorial purged CV, walk-forward CV,
and the deflated Sharpe ratio correction for multiple testing.

References:
  - Lopez de Prado, "Advances in Financial Machine Learning" (2018)
  - Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" (2014)
"""

import math
import itertools
import warnings
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import comb


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _embargo_start(test_start: int, embargo_bars: int, n: int) -> int:
    """Return the first training index allowed after a test block ends."""
    return min(test_start + embargo_bars, n)


def _overlap(event_start: int, event_end: int,
             test_start: int, test_end: int) -> bool:
    """Return True if the event [event_start, event_end] overlaps [test_start, test_end]."""
    return event_start <= test_end and event_end >= test_start


# ---------------------------------------------------------------------------
# PurgedKFoldCV
# ---------------------------------------------------------------------------

class PurgedKFoldCV:
    """
    Purged k-fold cross-validation for financial time series.

    Removes observations near the train/test boundary to prevent leakage.
    Implements combinatorial purged CV (CPCV) from Lopez de Prado.

    Parameters
    ----------
    n_splits : int
        Number of folds (k).
    embargo_pct : float
        Fraction of total samples used as embargo gap between train/test.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not 0.0 <= embargo_pct < 0.5:
            raise ValueError("embargo_pct must be in [0, 0.5)")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.embargo_bars: Optional[int] = None  -- set during split()

    # ------------------------------------------------------------------
    # Core split logic
    # ------------------------------------------------------------------

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        event_times: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_idx, test_idx) pairs with purging and embargo applied.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y : ignored, kept for sklearn compatibility
        groups : ignored, kept for sklearn compatibility
        event_times : array of shape (n_samples,) containing the event end
            bar index for each observation.  If None, each observation is
            treated as a point event (end == start).
        """
        n = len(X)
        self.embargo_bars = max(1, int(n * self.embargo_pct))

        indices = np.arange(n)
        fold_size = n // self.n_splits

        -- build fold boundaries
        fold_starts = [i * fold_size for i in range(self.n_splits)]
        fold_ends = [
            fold_starts[i + 1] if i + 1 < self.n_splits else n
            for i in range(self.n_splits)
        ]

        for fold_idx in range(self.n_splits):
            test_start = fold_starts[fold_idx]
            test_end = fold_ends[fold_idx] - 1

            test_idx = indices[test_start:test_end + 1]

            -- collect raw train candidates (everything outside the test fold)
            train_candidates = np.concatenate([
                indices[:test_start],
                indices[test_end + 1:],
            ])

            -- purge: remove train observations whose event spans overlap test
            if event_times is not None:
                train_candidates = self._purge(
                    train_candidates, event_times, test_start, test_end
                )

            -- embargo: remove train observations within embargo_bars of test_start
            train_idx = self._apply_embargo(train_candidates, test_start)

            yield train_idx, test_idx

    # ------------------------------------------------------------------
    # Purging helper
    # ------------------------------------------------------------------

    def _purge(
        self,
        train_candidates: np.ndarray,
        event_times: np.ndarray,
        test_start: int,
        test_end: int,
    ) -> np.ndarray:
        """
        Remove from train_candidates any index i where the event
        [i, event_times[i]] overlaps [test_start, test_end].
        """
        keep = []
        for i in train_candidates:
            ev_end = int(event_times[i])
            if not _overlap(i, ev_end, test_start, test_end):
                keep.append(i)
        return np.array(keep, dtype=int)

    # ------------------------------------------------------------------
    # Embargo helper
    # ------------------------------------------------------------------

    def _apply_embargo(
        self, train_candidates: np.ndarray, test_start: int
    ) -> np.ndarray:
        """
        Drop indices in [test_start - embargo_bars, test_start - 1] from train.
        We want to keep indices that are *before* the embargo window or
        *after* the test fold.
        """
        embargo = self.embargo_bars
        embargo_low = test_start - embargo
        embargo_high = test_start - 1

        mask = ~(
            (train_candidates >= embargo_low) & (train_candidates <= embargo_high)
        )
        return train_candidates[mask]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def __repr__(self) -> str:
        return (
            f"PurgedKFoldCV(n_splits={self.n_splits}, "
            f"embargo_pct={self.embargo_pct})"
        )


# ---------------------------------------------------------------------------
# CombinatorialPurgedCV
# ---------------------------------------------------------------------------

class CombinatorialPurgedCV:
    """
    Combinatorial purged cross-validation (CPCV).

    Tests on all C(n_splits, n_test_splits) combinations of folds.
    Yields (path_id, train_idx, test_idx) for each combination.

    This exhausts all unique paths through the data, giving a richer
    distribution of backtest results than standard k-fold.

    Parameters
    ----------
    n_splits : int
        Total number of folds (k).
    n_test_splits : int
        Number of folds held out as test in each combination (t).
    embargo_pct : float
        Fraction of total bars used as embargo.
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        embargo_pct: float = 0.01,
    ):
        if n_test_splits >= n_splits:
            raise ValueError("n_test_splits must be < n_splits")
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct
        self._base_cv = PurgedKFoldCV(
            n_splits=n_splits, embargo_pct=embargo_pct
        )

    @property
    def n_combinations(self) -> int:
        """Total number of (train, test) splits that will be generated."""
        return int(comb(self.n_splits, self.n_test_splits, exact=True))

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        event_times: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        """
        Yield (path_id, train_idx, test_idx) for every fold combination.
        """
        n = len(X)
        embargo_bars = max(1, int(n * self.embargo_pct))
        fold_size = n // self.n_splits

        fold_starts = [i * fold_size for i in range(self.n_splits)]
        fold_ends = [
            fold_starts[i + 1] if i + 1 < self.n_splits else n
            for i in range(self.n_splits)
        ]

        fold_indices = [
            np.arange(fold_starts[k], fold_ends[k])
            for k in range(self.n_splits)
        ]

        combo_iter = itertools.combinations(
            range(self.n_splits), self.n_test_splits
        )

        for path_id, test_folds in enumerate(combo_iter):
            test_set = set(test_folds)
            train_folds = [k for k in range(self.n_splits) if k not in test_set]

            test_idx = np.concatenate([fold_indices[k] for k in test_folds])
            train_candidates = np.concatenate(
                [fold_indices[k] for k in train_folds]
            ) if train_folds else np.array([], dtype=int)

            test_start = int(test_idx.min())
            test_end = int(test_idx.max())

            -- purge if event times provided
            if event_times is not None and len(train_candidates) > 0:
                train_candidates = self._base_cv._purge(
                    train_candidates, event_times, test_start, test_end
                )

            -- embargo around each test fold boundary
            if len(train_candidates) > 0:
                for tf in test_folds:
                    ts = fold_starts[tf]
                    train_candidates = self._base_cv._apply_embargo(
                        train_candidates, ts
                    )

            yield path_id, train_candidates, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_combinations

    def __repr__(self) -> str:
        return (
            f"CombinatorialPurgedCV(n_splits={self.n_splits}, "
            f"n_test_splits={self.n_test_splits}, "
            f"n_combinations={self.n_combinations})"
        )


# ---------------------------------------------------------------------------
# WalkForwardCV
# ---------------------------------------------------------------------------

class WalkForwardCV:
    """
    Walk-forward cross-validation for time series.

    Supports both expanding and rolling training windows.

    Parameters
    ----------
    n_initial : int
        Minimum number of training observations before the first test window.
    step_size : int
        Number of bars to advance the test window at each step.
    horizon : int
        Number of bars in each test window.
    rolling : bool
        If True, use a fixed-length rolling training window.
        If False (default), use an expanding window.
    max_train_size : int or None
        Maximum training window size when rolling=True.
    """

    def __init__(
        self,
        n_initial: int = 252,
        step_size: int = 21,
        horizon: int = 21,
        rolling: bool = False,
        max_train_size: Optional[int] = None,
    ):
        if n_initial < 1:
            raise ValueError("n_initial must be >= 1")
        if step_size < 1:
            raise ValueError("step_size must be >= 1")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.n_initial = n_initial
        self.step_size = step_size
        self.horizon = horizon
        self.rolling = rolling
        self.max_train_size = max_train_size

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_idx, test_idx) for each walk-forward step.
        """
        n = len(X)
        indices = np.arange(n)

        test_start = self.n_initial

        while test_start + self.horizon <= n:
            test_end = test_start + self.horizon

            test_idx = indices[test_start:test_end]

            if self.rolling and self.max_train_size is not None:
                train_start = max(0, test_start - self.max_train_size)
            else:
                train_start = 0

            train_idx = indices[train_start:test_start]

            yield train_idx, test_idx

            test_start += self.step_size

    def get_n_splits(self, X: np.ndarray, y=None, groups=None) -> int:
        n = len(X)
        count = 0
        test_start = self.n_initial
        while test_start + self.horizon <= n:
            count += 1
            test_start += self.step_size
        return count

    def __repr__(self) -> str:
        mode = "rolling" if self.rolling else "expanding"
        return (
            f"WalkForwardCV(n_initial={self.n_initial}, "
            f"step_size={self.step_size}, horizon={self.horizon}, "
            f"mode={mode})"
        )


# ---------------------------------------------------------------------------
# DeflatedSharpeRatio
# ---------------------------------------------------------------------------

def DeflatedSharpeRatio(
    sharpe: float,
    n_trials: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    n_obs: int = 252,
) -> float:
    """
    Compute the deflated Sharpe ratio (DSR) from Bailey & Lopez de Prado (2014).

    Corrects the observed Sharpe ratio for the bias introduced by evaluating
    multiple strategies/configurations and selecting the best.

    Parameters
    ----------
    sharpe : float
        Observed annualised Sharpe ratio of the selected strategy.
    n_trials : int
        Number of independent trials / strategies tested.
    skew : float
        Skewness of strategy returns (0 = normal).
    kurtosis : float
        Excess kurtosis of strategy returns (3 = normal, i.e. excess = 0).
    n_obs : int
        Number of observations used to estimate the Sharpe ratio.

    Returns
    -------
    float
        Probability that the true Sharpe ratio is positive (DSR p-value).
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be >= 1")
    if n_obs <= 4:
        raise ValueError("n_obs must be > 4")

    -- expected maximum Sharpe ratio under IID normal trials
    -- approximation: E[max SR] ~ (1 - gamma) * Z^{-1}(1 - 1/N) + gamma * Z^{-1}(1 - 1/Ne)
    -- simplified Euler-Mascheroni form used by Bailey & Lopez de Prado
    emc = 0.5772156649  -- Euler-Mascheroni constant

    if n_trials == 1:
        sr_star = 0.0
    else:
        -- expected max of n_trials IID standard normals
        z_quantile = stats.norm.ppf(1.0 - 1.0 / n_trials)
        sr_star = (
            (1.0 - emc) * stats.norm.ppf(1.0 - 1.0 / n_trials)
            + emc * stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        )

    -- variance of the Sharpe ratio estimate (non-normal correction)
    -- var(SR_hat) = (1 + 0.5*SR^2 - skew*SR + (kurtosis-3)/4 * SR^2) / (T-1)
    excess_kurt = kurtosis - 3.0
    sr_variance = (
        1.0
        + 0.5 * sharpe ** 2
        - skew * sharpe
        + (excess_kurt / 4.0) * sharpe ** 2
    ) / (n_obs - 1)

    sr_std = math.sqrt(max(sr_variance, 1e-12))

    -- DSR: probability that the strategy SR beats the expected max
    dsr = stats.norm.cdf((sharpe - sr_star) / sr_std)
    return float(dsr)


# ---------------------------------------------------------------------------
# MinTRL -- minimum track record length
# ---------------------------------------------------------------------------

def MinimumTrackRecordLength(
    sharpe: float,
    sr_benchmark: float = 0.0,
    alpha: float = 0.05,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Compute the minimum number of observations needed to reject H0: SR <= sr_benchmark
    at significance level alpha, accounting for non-normality.

    Parameters
    ----------
    sharpe : float
        Observed annualised Sharpe ratio.
    sr_benchmark : float
        Benchmark Sharpe ratio under H0 (usually 0).
    alpha : float
        Significance level (e.g. 0.05).
    skew : float
        Skewness of returns.
    kurtosis : float
        Excess kurtosis of returns.

    Returns
    -------
    float
        Minimum track record length in observations.
    """
    if sharpe <= sr_benchmark:
        return float("inf")

    z_alpha = stats.norm.ppf(1.0 - alpha)
    excess_kurt = kurtosis - 3.0

    -- estimate variance multiplier for non-normal returns
    var_mult = (
        1.0
        + 0.5 * sharpe ** 2
        - skew * sharpe
        + (excess_kurt / 4.0) * sharpe ** 2
    )

    min_trl = 1.0 + var_mult * (z_alpha / (sharpe - sr_benchmark)) ** 2
    return float(min_trl)


# ---------------------------------------------------------------------------
# CrossValidationReport
# ---------------------------------------------------------------------------

class CrossValidationReport:
    """
    Aggregates metrics across CV folds and provides summary statistics.

    Parameters
    ----------
    cv_results : list of dict
        Each dict must contain at least {'fold': int, 'val_score': float}.
    """

    def __init__(self, cv_results: List[dict]):
        self.cv_results = cv_results
        self._df = pd.DataFrame(cv_results)

    @property
    def mean_score(self) -> float:
        return float(self._df["val_score"].mean())

    @property
    def std_score(self) -> float:
        return float(self._df["val_score"].std())

    @property
    def sharpe_of_folds(self) -> float:
        """Sharpe ratio of per-fold validation scores."""
        mu = self._df["val_score"].mean()
        sigma = self._df["val_score"].std()
        if sigma < 1e-10:
            return 0.0
        return float(mu / sigma)

    def summary(self) -> pd.DataFrame:
        """Return a one-row summary DataFrame."""
        return pd.DataFrame(
            {
                "mean_score": [self.mean_score],
                "std_score": [self.std_score],
                "min_score": [self._df["val_score"].min()],
                "max_score": [self._df["val_score"].max()],
                "n_folds": [len(self._df)],
                "sharpe_of_folds": [self.sharpe_of_folds],
            }
        )

    def fold_scores(self) -> pd.Series:
        return self._df.set_index("fold")["val_score"]

    def __repr__(self) -> str:
        return (
            f"CrossValidationReport("
            f"n_folds={len(self._df)}, "
            f"mean={self.mean_score:.4f}, "
            f"std={self.std_score:.4f})"
        )


# ---------------------------------------------------------------------------
# Utility: compute_information_coefficient
# ---------------------------------------------------------------------------

def compute_information_coefficient(
    y_pred: np.ndarray, y_true: np.ndarray
) -> float:
    """
    Compute Spearman rank IC between predictions and realizations.

    Returns float in [-1, 1].  NaN values are dropped pairwise.
    """
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    if mask.sum() < 3:
        return float("nan")
    ic, _ = stats.spearmanr(y_pred[mask], y_true[mask])
    return float(ic)


# ---------------------------------------------------------------------------
# Utility: cv_score convenience function
# ---------------------------------------------------------------------------

def cv_score(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: PurgedKFoldCV,
    scoring_fn=None,
    event_times: Optional[np.ndarray] = None,
) -> CrossValidationReport:
    """
    Run purged cross-validation and collect per-fold scores.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    cv : PurgedKFoldCV instance
    scoring_fn : callable(y_true, y_pred) -> float, defaults to IC
    event_times : optional event end times for purging

    Returns
    -------
    CrossValidationReport
    """
    if scoring_fn is None:
        scoring_fn = compute_information_coefficient

    results = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        cv.split(X, y, event_times=event_times)
    ):
        if len(train_idx) == 0 or len(test_idx) == 0:
            warnings.warn(f"Fold {fold_idx} has empty train or test set, skipping.")
            continue

        estimator.fit(X[train_idx], y[train_idx])
        y_pred = estimator.predict(X[test_idx])
        score = scoring_fn(y_pred, y[test_idx])

        results.append(
            {
                "fold": fold_idx,
                "val_score": score,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        )

    return CrossValidationReport(results)
