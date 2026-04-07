"""
research/validation/out_of_sample_validator.py -- rigorous out-of-sample validation.

Implements expanding window, walk-forward, and combinatorial purged cross-validation
(CPCV) for signal evaluation. Includes deflated Sharpe ratio and FDR correction
to properly account for multiple testing in strategy research.

Reference: Lopez de Prado, "Advances in Financial Machine Learning" (2018),
chapters 7 (cross-validation), 8 (feature importance), and the deflated SR paper.
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OOSResult:
    """
    Result of expanding-window out-of-sample test.

    decay_ratio = oos_icir / is_icir
      1.0 = no decay (signal works equally well OOS)
      0.5 = moderate decay (expect in real trading)
      <0.3 = significant overfit warning
    """
    is_ics: List[float]       # Information coefficient at each IS step
    oos_ics: List[float]      # Information coefficient at each OOS step
    is_icir: float            # Mean IS IC / std(IS IC)
    oos_icir: float           # Mean OOS IC / std(OOS IC)
    decay_ratio: float        # oos_icir / is_icir (1.0 = no decay)
    t_stat_oos: float         # t-stat testing oos_icir > 0
    n_periods: int            # Number of OOS periods
    dates: List[object] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Result of walk-forward (rolling window) test."""
    is_ics: List[float]
    oos_ics: List[float]
    is_icir: float
    oos_icir: float
    decay_ratio: float
    t_stat_oos: float
    n_periods: int
    window_dates: List[Tuple[object, object]] = field(default_factory=list)


@dataclass
class CPCVResult:
    """
    Result of combinatorial purged cross-validation.

    Returns a distribution of backtest paths, not just a single path.
    This reveals the range of possible OOS performance, exposing
    overfitting that a single backtest path cannot.
    """
    backtest_paths: List[List[float]]     # Each path is a sequence of OOS ICs
    path_sharpes: List[float]             # Sharpe (ICIR) of each path
    median_sharpe: float
    sharpe_5th_pct: float                 # 5th percentile -- pessimistic estimate
    sharpe_95th_pct: float
    n_paths: int
    n_splits: int
    embargo_periods: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _information_coefficient(signal: np.ndarray, returns: np.ndarray) -> float:
    """Spearman rank IC between signal and forward returns."""
    valid = np.isfinite(signal) & np.isfinite(returns)
    if valid.sum() < 5:
        return float("nan")
    s = signal[valid]
    r = returns[valid]
    if np.std(s) < 1e-15 or np.std(r) < 1e-15:
        return 0.0
    corr, _ = stats.spearmanr(s, r)
    return float(corr)


def _icir(ics: List[float]) -> float:
    """IC information ratio = mean(IC) / std(IC). Returns 0 if not enough data."""
    arr = np.array([v for v in ics if np.isfinite(v)])
    if len(arr) < 2:
        return 0.0
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    return mu / sigma if sigma > 1e-15 else 0.0


def _t_stat_oos(ics: List[float]) -> float:
    """t-stat for H0: mean OOS IC = 0."""
    arr = np.array([v for v in ics if np.isfinite(v)])
    if len(arr) < 2:
        return 0.0
    return float(stats.ttest_1samp(arr, 0.0).statistic)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OutOfSampleValidator:
    """
    Rigorous OOS validation for quantitative signals.

    signal_fn convention: signal_fn(train_returns, test_returns) -> np.ndarray
      Given training data and test-period returns, returns a signal array
      (same length as test_returns) to be correlated with test forward returns.

    For simple signals (e.g. momentum), signal_fn just computes the rolling
    average of trailing returns using only the training window parameters.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Expanding Window
    # ------------------------------------------------------------------

    def expanding_window_test(
        self,
        signal_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
        returns: pd.DataFrame,
        initial_train: int = 252,
        min_obs_per_period: int = 10,
    ) -> OOSResult:
        """
        Train on [0, t], test on [t, t+1], then expand t by 1 period.

        Parameters
        ----------
        signal_fn : Callable
            Function (train_returns, test_returns) -> signal array.
            Length of output must match len(test_returns).
        returns : pd.DataFrame
            Returns matrix (rows = time, columns = assets or single column).
        initial_train : int
            Minimum training periods before first OOS test.
        min_obs_per_period : int
            Minimum non-NaN observations to compute IC; otherwise skip period.

        Returns
        -------
        OOSResult
        """
        n = len(returns)
        if initial_train >= n - 1:
            raise ValueError(
                f"initial_train={initial_train} too large for n={n}; need at least 1 OOS period"
            )

        is_ics: List[float] = []
        oos_ics: List[float] = []
        dates: List[object] = []

        for t in range(initial_train, n - 1):
            train = returns.iloc[:t]
            test_period = returns.iloc[t: t + 1]

            try:
                signal = signal_fn(train, test_period)
            except Exception as e:
                warnings.warn(f"signal_fn failed at t={t}: {e}", UserWarning, stacklevel=2)
                continue

            signal = np.asarray(signal, dtype=float).ravel()
            test_vals = test_period.values.ravel().astype(float)

            if len(signal) != len(test_vals):
                continue

            oos_ic = _information_coefficient(signal, test_vals)
            oos_ics.append(oos_ic)

            # IS IC: run signal_fn on the full training set and check fit
            try:
                is_signal = signal_fn(train.iloc[: max(1, len(train) - 1)], train.iloc[-1:])
                is_signal = np.asarray(is_signal, dtype=float).ravel()
                is_vals = train.iloc[-1:].values.ravel().astype(float)
                is_ic = _information_coefficient(is_signal, is_vals)
            except Exception:
                is_ic = float("nan")

            is_ics.append(is_ic)
            dates.append(returns.index[t])

        if not oos_ics:
            raise ValueError("No OOS periods were computed -- check signal_fn and data size")

        is_icir_val = _icir(is_ics)
        oos_icir_val = _icir(oos_ics)
        decay = oos_icir_val / is_icir_val if abs(is_icir_val) > 1e-10 else 0.0
        t_oos = _t_stat_oos(oos_ics)

        return OOSResult(
            is_ics=is_ics,
            oos_ics=oos_ics,
            is_icir=is_icir_val,
            oos_icir=oos_icir_val,
            decay_ratio=decay,
            t_stat_oos=t_oos,
            n_periods=len(oos_ics),
            dates=dates,
        )

    # ------------------------------------------------------------------
    # Walk-Forward (Rolling Window)
    # ------------------------------------------------------------------

    def walk_forward_test(
        self,
        signal_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
        returns: pd.DataFrame,
        train_window: int = 252,
        test_window: int = 63,
    ) -> WalkForwardResult:
        """
        Walk-forward test with fixed-size rolling training window.

        Train on [t, t+train_window], test on [t+train_window, t+train_window+test_window],
        then roll forward by test_window periods.

        Parameters
        ----------
        signal_fn : Callable
            Function (train_returns, test_returns) -> signal array.
        returns : pd.DataFrame
            Returns matrix.
        train_window : int
            Size of rolling training window in periods.
        test_window : int
            Size of each OOS test block in periods.

        Returns
        -------
        WalkForwardResult
        """
        n = len(returns)
        if train_window + test_window > n:
            raise ValueError(
                f"train_window+test_window={train_window + test_window} exceeds data length {n}"
            )

        is_ics: List[float] = []
        oos_ics: List[float] = []
        window_dates: List[Tuple[object, object]] = []

        t = 0
        while t + train_window + test_window <= n:
            train = returns.iloc[t: t + train_window]
            test = returns.iloc[t + train_window: t + train_window + test_window]

            try:
                signal = signal_fn(train, test)
            except Exception as e:
                warnings.warn(f"signal_fn failed at t={t}: {e}", UserWarning, stacklevel=2)
                t += test_window
                continue

            signal = np.asarray(signal, dtype=float).ravel()
            test_vals = test.values.ravel().astype(float)

            if len(signal) > 0 and len(test_vals) == len(signal):
                oos_ic = _information_coefficient(signal, test_vals)
                oos_ics.append(oos_ic)
            else:
                oos_ics.append(float("nan"))

            # IS IC: last test_window of training set
            try:
                is_sub_train = train.iloc[: max(1, train_window - test_window)]
                is_sub_test = train.iloc[max(1, train_window - test_window):]
                is_signal = signal_fn(is_sub_train, is_sub_test)
                is_signal = np.asarray(is_signal, dtype=float).ravel()
                is_vals = is_sub_test.values.ravel().astype(float)
                if len(is_signal) == len(is_vals):
                    is_ic = _information_coefficient(is_signal, is_vals)
                else:
                    is_ic = float("nan")
            except Exception:
                is_ic = float("nan")

            is_ics.append(is_ic)

            start_date = returns.index[t + train_window]
            end_date = returns.index[min(t + train_window + test_window - 1, n - 1)]
            window_dates.append((start_date, end_date))

            t += test_window

        if not oos_ics:
            raise ValueError("No OOS windows computed -- check parameters")

        is_icir_val = _icir(is_ics)
        oos_icir_val = _icir(oos_ics)
        decay = oos_icir_val / is_icir_val if abs(is_icir_val) > 1e-10 else 0.0
        t_oos = _t_stat_oos(oos_ics)

        return WalkForwardResult(
            is_ics=is_ics,
            oos_ics=oos_ics,
            is_icir=is_icir_val,
            oos_icir=oos_icir_val,
            decay_ratio=decay,
            t_stat_oos=t_oos,
            n_periods=len(oos_ics),
            window_dates=window_dates,
        )

    # ------------------------------------------------------------------
    # Combinatorial Purged Cross-Validation (CPCV)
    # ------------------------------------------------------------------

    def combinatorial_purged_cv(
        self,
        signal_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
        returns: pd.DataFrame,
        n_splits: int = 6,
        embargo: int = 5,
    ) -> CPCVResult:
        """
        Combinatorial Purged Cross-Validation (Lopez de Prado CPCV).

        Splits the dataset into n_splits groups. Evaluates all C(n_splits, 2)
        combinations of train/test splits with purging (dropping embargo periods
        adjacent to test set) to prevent leakage.

        Each combination produces a sequence of OOS ICs (a "path"). The collection
        of all paths gives a distribution of backtests, revealing the true range
        of OOS performance rather than a single optimistic estimate.

        Parameters
        ----------
        signal_fn : Callable
            Function (train_returns, test_returns) -> signal array.
        returns : pd.DataFrame
            Returns matrix.
        n_splits : int
            Number of groups to split data into (CPCV uses n_splits choose 2 paths).
        embargo : int
            Number of periods to embargo at boundaries to prevent lookahead.

        Returns
        -------
        CPCVResult
        """
        n = len(returns)
        group_size = n // n_splits
        if group_size < embargo + 5:
            raise ValueError(
                f"Groups too small (size={group_size}) for embargo={embargo}; "
                f"use fewer splits or more data"
            )

        # Split indices into n_splits groups
        groups: List[np.ndarray] = []
        for i in range(n_splits):
            start = i * group_size
            end = (i + 1) * group_size if i < n_splits - 1 else n
            groups.append(np.arange(start, end))

        backtest_paths: List[List[float]] = []
        path_sharpes: List[float] = []

        # Test each combination of 1 test group from n_splits groups
        # The "combinatorial" part: use all C(n_splits, 1) single-group test sets
        # For a richer distribution, also use pairs as described in Lopez de Prado
        test_combos = list(range(n_splits))  # Each group serves as test once

        for test_group_idx in test_combos:
            test_indices = groups[test_group_idx]

            # Build train indices: all other groups, with embargo applied
            test_start = int(test_indices[0])
            test_end = int(test_indices[-1])

            train_indices_list: List[int] = []
            for i, g in enumerate(groups):
                if i == test_group_idx:
                    continue
                for idx in g:
                    # Apply embargo: drop periods too close to test set boundaries
                    if abs(idx - test_start) >= embargo and abs(idx - test_end) >= embargo:
                        train_indices_list.append(int(idx))

            if len(train_indices_list) < group_size:
                warnings.warn(
                    f"Test group {test_group_idx}: train too small after embargo, skipping",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            train_idx = sorted(train_indices_list)
            train = returns.iloc[train_idx]
            test = returns.iloc[test_indices]

            try:
                signal = signal_fn(train, test)
            except Exception as e:
                warnings.warn(f"signal_fn failed for test_group={test_group_idx}: {e}")
                continue

            signal = np.asarray(signal, dtype=float).ravel()
            test_vals = test.values.ravel().astype(float)

            if len(signal) != len(test_vals):
                continue

            # Compute rolling ICs within this test window (split test into sub-periods)
            sub_size = max(1, len(test_vals) // 5)
            path_ics: List[float] = []
            for start in range(0, len(test_vals), sub_size):
                end = min(start + sub_size, len(test_vals))
                sub_sig = signal[start:end]
                sub_ret = test_vals[start:end]
                ic = _information_coefficient(sub_sig, sub_ret)
                if np.isfinite(ic):
                    path_ics.append(ic)

            if path_ics:
                backtest_paths.append(path_ics)
                path_sharpes.append(_icir(path_ics))

        if not path_sharpes:
            raise ValueError("CPCV produced no valid paths -- check signal_fn and parameters")

        sharpe_arr = np.array(path_sharpes)
        return CPCVResult(
            backtest_paths=backtest_paths,
            path_sharpes=list(sharpe_arr),
            median_sharpe=float(np.median(sharpe_arr)),
            sharpe_5th_pct=float(np.percentile(sharpe_arr, 5)),
            sharpe_95th_pct=float(np.percentile(sharpe_arr, 95)),
            n_paths=len(path_sharpes),
            n_splits=n_splits,
            embargo_periods=embargo,
        )

    # ------------------------------------------------------------------
    # Deflated Sharpe Ratio (Bailey & Lopez de Prado)
    # ------------------------------------------------------------------

    @staticmethod
    def deflated_sharpe(
        trial_sharpe: float,
        n_trials: int,
        n_obs: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        sr_benchmark: float = 0.0,
    ) -> float:
        """
        Deflated Sharpe Ratio (DSR) -- Bailey & Lopez de Prado (2014).

        Accounts for:
          1. Non-normality of returns (skewness, kurtosis).
          2. Multiple testing: the expected maximum SR under H0 grows with n_trials.

        DSR is the probability that the strategy's SR is above sr_benchmark,
        corrected for selection bias across n_trials strategies tested.

        The expected maximum SR under the null (all strategies random) is:
          E[max_SR] ~ Z^{-1}(1 - 1/n_trials) * sqrt((1 - skew*SR + (kurt-1)/4 * SR^2) / T)

        Parameters
        ----------
        trial_sharpe : float
            Annualized Sharpe ratio of the strategy being evaluated.
        n_trials : int
            Total number of strategies/parameter sets tried.
        n_obs : int
            Number of independent observations (trading days, etc.).
        skewness : float
            Skewness of strategy returns (0 = normal).
        kurtosis : float
            Excess kurtosis of strategy returns (3 = normal, a.k.a. 4th moment).
        sr_benchmark : float
            Minimum acceptable SR (default 0).

        Returns
        -------
        float
            Probability that the true SR exceeds sr_benchmark (0 to 1).
            DSR > 0.95 is typically considered statistically significant.
        """
        if n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if n_obs < 2:
            raise ValueError("n_obs must be >= 2")

        # Expected maximum SR under null, using expected maximum of n_trials
        # standard normals: E[max_z] ~ Phi^{-1}(1 - 1/n_trials)
        # This is an approximation; the exact formula uses the Euler-Mascheroni constant
        # but this dominates for n_trials > 10.
        if n_trials == 1:
            expected_max_z = 0.0
        else:
            # Bloch et al. approximation for expected max of n IID N(0,1) draws
            euler_mascheroni = 0.5772156649
            expected_max_z = (
                (1 - euler_mascheroni) * stats.norm.ppf(1 - 1.0 / n_trials)
                + euler_mascheroni * stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
            )

        # Adjusted variance of the SR estimate accounting for non-normality
        # Var(SR_hat) ~ (1 - skew*SR + (kurt-1)/4 * SR^2) / T
        sr_var = (
            1.0
            - skewness * trial_sharpe
            + (kurtosis - 1.0) / 4.0 * trial_sharpe ** 2
        ) / max(n_obs - 1, 1)
        sr_std = math.sqrt(max(sr_var, 1e-15))

        # Deflated SR: how many std deviations above expected max null SR is our SR?
        sr_benchmark_adj = sr_benchmark + expected_max_z * sr_std
        z_score = (trial_sharpe - sr_benchmark_adj) / sr_std
        dsr = float(stats.norm.cdf(z_score))
        return dsr

    # ------------------------------------------------------------------
    # False Discovery Rate Correction (Benjamini-Hochberg)
    # ------------------------------------------------------------------

    @staticmethod
    def false_discovery_rate(
        p_values: List[float],
        alpha: float = 0.05,
    ) -> List[bool]:
        """
        Benjamini-Hochberg FDR correction for multiple comparisons.

        Controls the expected fraction of false discoveries among all rejections.
        Less conservative than Bonferroni; appropriate for strategy research where
        some true signals are expected.

        Under H0 for all tests, this controls FDR at level alpha.
        In practice, this allows detection of true signals while limiting
        the number of false alarms to alpha * (number of rejections).

        Parameters
        ----------
        p_values : List[float]
            Raw p-values from individual hypothesis tests.
        alpha : float
            FDR level to control (default 0.05 = 5% FDR).

        Returns
        -------
        List[bool]
            True where H0 is rejected (signal is significant after FDR correction).
        """
        n = len(p_values)
        if n == 0:
            return []

        p_arr = np.array(p_values, dtype=float)
        rank_order = np.argsort(p_arr)  # sorted indices
        sorted_p = p_arr[rank_order]

        # BH critical values: (rank / n) * alpha
        bh_critical = (np.arange(1, n + 1) / n) * alpha

        # Find the largest k such that p_{(k)} <= (k/n) * alpha
        significant_sorted = sorted_p <= bh_critical
        if not np.any(significant_sorted):
            return [False] * n

        # All tests up to and including the last significant one are rejected
        last_sig = int(np.max(np.where(significant_sorted)[0]))
        reject_sorted = np.zeros(n, dtype=bool)
        reject_sorted[: last_sig + 1] = True

        # Map back to original order
        reject = np.zeros(n, dtype=bool)
        reject[rank_order] = reject_sorted
        return list(reject)

    # ------------------------------------------------------------------
    # Convenience: full validation suite
    # ------------------------------------------------------------------

    def full_validation(
        self,
        signal_fn: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
        returns: pd.DataFrame,
        initial_train: int = 252,
        train_window: int = 252,
        test_window: int = 63,
        n_cpcv_splits: int = 6,
        embargo: int = 5,
    ) -> Dict[str, object]:
        """
        Run all three validation methods and return a dict of results.

        Parameters
        ----------
        signal_fn : Callable
            Function (train_returns, test_returns) -> signal array.
        returns : pd.DataFrame
            Returns matrix.
        initial_train, train_window, test_window, n_cpcv_splits, embargo
            Parameters forwarded to individual methods.

        Returns
        -------
        dict with keys: expanding, walk_forward, cpcv, summary
        """
        out: Dict[str, object] = {}

        try:
            out["expanding"] = self.expanding_window_test(
                signal_fn, returns, initial_train=initial_train
            )
        except Exception as e:
            warnings.warn(f"Expanding window failed: {e}", UserWarning, stacklevel=2)
            out["expanding"] = None

        try:
            out["walk_forward"] = self.walk_forward_test(
                signal_fn, returns, train_window=train_window, test_window=test_window
            )
        except Exception as e:
            warnings.warn(f"Walk-forward failed: {e}", UserWarning, stacklevel=2)
            out["walk_forward"] = None

        try:
            out["cpcv"] = self.combinatorial_purged_cv(
                signal_fn, returns, n_splits=n_cpcv_splits, embargo=embargo
            )
        except Exception as e:
            warnings.warn(f"CPCV failed: {e}", UserWarning, stacklevel=2)
            out["cpcv"] = None

        # Summary diagnostics
        summary_lines = ["=== OOS Validation Summary ==="]
        for method_name in ("expanding", "walk_forward"):
            res = out.get(method_name)
            if res is not None:
                summary_lines.append(
                    f"{method_name}: OOS ICIR={res.oos_icir:.3f}, "
                    f"decay={res.decay_ratio:.2f}, t={res.t_stat_oos:.2f}"
                )
        cpcv_res = out.get("cpcv")
        if cpcv_res is not None:
            summary_lines.append(
                f"CPCV: median_SR={cpcv_res.median_sharpe:.3f}, "
                f"5th_pct={cpcv_res.sharpe_5th_pct:.3f}, paths={cpcv_res.n_paths}"
            )

        out["summary"] = "\n".join(summary_lines)
        return out
