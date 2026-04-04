"""
research/signal_analytics/diagnostics.py
==========================================
Signal diagnostics: statistical tests, overfitting checks, and
robustness analysis for the SRFM-Lab signal analytics framework.

Provides:
  - Sharpe ratio significance tests
  - Multiple-testing correction (Bonferroni, BH)
  - Walk-forward signal validation
  - Permutation test for IC significance
  - Signal overfitting diagnostics
  - Return attribution decomposition
  - Regime-conditioned performance breakdown
  - Statistical arbitrage tests (cointegration, lead-lag)

Usage example
-------------
>>> diag = SignalDiagnostics()
>>> perm_result = diag.permutation_ic_test(signal, returns, n_permutations=1000)
>>> overfitting = diag.overfitting_diagnosis(trades, signal_col, n_splits=5)
>>> diag.plot_walk_forward_ic(wf_result, save_path="results/wf_ic.png")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PermutationTestResult:
    """Result of a permutation/bootstrap significance test."""
    observed_ic: float
    p_value: float
    null_distribution: np.ndarray
    ci_lower: float
    ci_upper: float
    n_permutations: int
    is_significant: bool
    confidence_level: float = 0.95


@dataclass
class WalkForwardResult:
    """Walk-forward validation result."""
    n_splits: int
    train_ics: List[float]
    test_ics: List[float]
    train_sizes: List[int]
    test_sizes: List[int]
    mean_train_ic: float
    mean_test_ic: float
    ic_degradation: float       # train_ic - test_ic
    overfitting_ratio: float    # test_ic / train_ic


@dataclass
class OverfittingDiagnosis:
    """Signal overfitting diagnosis."""
    n_splits: int
    mean_is_ic: float           # in-sample IC
    mean_oos_ic: float          # out-of-sample IC
    ic_degradation: float
    overfitting_score: float    # 0 = no overfitting, 1 = severe
    is_significant_oos: bool
    recommendation: str


@dataclass
class MultipleTestingResult:
    """Multiple hypothesis testing correction results."""
    raw_p_values: Dict[str, float]
    bonferroni_p: Dict[str, float]
    bh_p: Dict[str, float]
    significant_bonferroni: Dict[str, bool]
    significant_bh: Dict[str, bool]
    alpha: float
    n_tests: int


# ---------------------------------------------------------------------------
# SignalDiagnostics
# ---------------------------------------------------------------------------

class SignalDiagnostics:
    """Statistical diagnostic tools for signal validation.

    Parameters
    ----------
    alpha : significance level (default 0.05)
    seed  : random seed for permutation tests
    """

    def __init__(self, alpha: float = 0.05, seed: int = 42) -> None:
        self.alpha = alpha
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Permutation IC test
    # ------------------------------------------------------------------ #

    def permutation_ic_test(
        self,
        signal: pd.Series,
        forward_returns: pd.Series,
        n_permutations: int = 1000,
        method: str = "spearman",
        confidence: float = 0.95,
    ) -> PermutationTestResult:
        """Non-parametric permutation test for IC significance.

        Under H0: signal and returns are independent, permute signal
        labels and compute IC distribution.

        Parameters
        ----------
        signal          : signal values
        forward_returns : forward returns
        n_permutations  : number of permutations
        method          : 'spearman' or 'pearson'
        confidence      : confidence level for CI

        Returns
        -------
        PermutationTestResult
        """
        df = pd.concat({"sig": signal, "ret": forward_returns}, axis=1).dropna()
        if len(df) < 5:
            return PermutationTestResult(
                observed_ic=float("nan"), p_value=float("nan"),
                null_distribution=np.array([]), ci_lower=float("nan"),
                ci_upper=float("nan"), n_permutations=0, is_significant=False,
            )

        sig_vals = df["sig"].values
        ret_vals = df["ret"].values

        # Observed IC
        if method == "spearman":
            obs_r, _ = stats.spearmanr(sig_vals, ret_vals)
        else:
            obs_r, _ = stats.pearsonr(sig_vals, ret_vals)
        obs_ic = float(obs_r)

        # Null distribution
        null_ics: list[float] = []
        for _ in range(n_permutations):
            perm_sig = self._rng.permutation(sig_vals)
            if method == "spearman":
                r, _ = stats.spearmanr(perm_sig, ret_vals)
            else:
                r, _ = stats.pearsonr(perm_sig, ret_vals)
            null_ics.append(float(r))

        null_arr = np.array(null_ics)
        # One-tailed p-value (IC > 0 means signal predicts returns)
        if obs_ic >= 0:
            p_val = float(np.mean(null_arr >= obs_ic))
        else:
            p_val = float(np.mean(null_arr <= obs_ic))

        alpha_half = (1 - confidence) / 2
        ci_lo = float(np.quantile(null_arr, alpha_half))
        ci_hi = float(np.quantile(null_arr, 1 - alpha_half))

        return PermutationTestResult(
            observed_ic=obs_ic,
            p_value=p_val,
            null_distribution=null_arr,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            n_permutations=n_permutations,
            is_significant=(p_val < self.alpha),
            confidence_level=confidence,
        )

    # ------------------------------------------------------------------ #
    # Walk-forward validation
    # ------------------------------------------------------------------ #

    def walk_forward_ic(
        self,
        trades: pd.DataFrame,
        signal_col: str,
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
        n_splits: int = 5,
        gap: int = 0,
        method: str = "spearman",
    ) -> WalkForwardResult:
        """Walk-forward IC validation: train on past, test on future.

        Chronologically splits data into n_splits folds.
        Each fold trains on data up to the split point and tests on
        the next segment.

        Parameters
        ----------
        trades        : trade records (sorted chronologically if exit_time present)
        signal_col    : signal column
        return_col    : P&L column
        dollar_pos_col: position column
        n_splits      : number of train/test splits
        gap           : number of observations to skip between train and test
        method        : IC correlation method

        Returns
        -------
        WalkForwardResult
        """
        df = trades.copy()
        if "exit_time" in df.columns:
            df = df.sort_values("exit_time")

        if dollar_pos_col in df.columns:
            pos = df[dollar_pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[return_col] / pos
        else:
            df["_ret"] = df[return_col]

        sub = df[[signal_col, "_ret"]].dropna()
        n = len(sub)

        # Create expanding-window splits
        min_train = n // (n_splits + 1)
        fold_size = (n - min_train) // n_splits

        train_ics: list[float] = []
        test_ics: list[float] = []
        train_sizes: list[int] = []
        test_sizes: list[int] = []

        for i in range(n_splits):
            train_end = min_train + i * fold_size
            test_start = train_end + gap
            test_end = test_start + fold_size

            if test_end > n:
                break

            train = sub.iloc[:train_end]
            test = sub.iloc[test_start:test_end]

            if len(train) < 5 or len(test) < 5:
                continue

            if method == "spearman":
                r_tr, _ = stats.spearmanr(train[signal_col], train["_ret"])
                r_te, _ = stats.spearmanr(test[signal_col], test["_ret"])
            else:
                r_tr, _ = stats.pearsonr(train[signal_col], train["_ret"])
                r_te, _ = stats.pearsonr(test[signal_col], test["_ret"])

            train_ics.append(float(r_tr))
            test_ics.append(float(r_te))
            train_sizes.append(len(train))
            test_sizes.append(len(test))

        if not train_ics:
            return WalkForwardResult(
                n_splits=0, train_ics=[], test_ics=[], train_sizes=[], test_sizes=[],
                mean_train_ic=float("nan"), mean_test_ic=float("nan"),
                ic_degradation=float("nan"), overfitting_ratio=float("nan"),
            )

        mean_tr = float(np.nanmean(train_ics))
        mean_te = float(np.nanmean(test_ics))
        deg = mean_tr - mean_te
        oor = mean_te / mean_tr if mean_tr != 0 else float("nan")

        return WalkForwardResult(
            n_splits=len(train_ics),
            train_ics=train_ics,
            test_ics=test_ics,
            train_sizes=train_sizes,
            test_sizes=test_sizes,
            mean_train_ic=mean_tr,
            mean_test_ic=mean_te,
            ic_degradation=deg,
            overfitting_ratio=oor,
        )

    # ------------------------------------------------------------------ #
    # Overfitting diagnosis
    # ------------------------------------------------------------------ #

    def overfitting_diagnosis(
        self,
        trades: pd.DataFrame,
        signal_col: str,
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
        n_splits: int = 5,
    ) -> OverfittingDiagnosis:
        """Diagnose potential signal overfitting.

        Compares in-sample vs out-of-sample IC and provides a recommendation.

        Parameters
        ----------
        trades        : trade records
        signal_col    : signal column
        return_col    : P&L column
        dollar_pos_col: position column
        n_splits      : number of CV splits

        Returns
        -------
        OverfittingDiagnosis with recommendation string
        """
        wf = self.walk_forward_ic(
            trades, signal_col, return_col, dollar_pos_col, n_splits=n_splits
        )

        if not wf.test_ics:
            return OverfittingDiagnosis(
                n_splits=0, mean_is_ic=float("nan"), mean_oos_ic=float("nan"),
                ic_degradation=float("nan"), overfitting_score=float("nan"),
                is_significant_oos=False, recommendation="Insufficient data for diagnosis.",
            )

        # Overfitting score: 0 = no degradation, 1 = complete degradation
        if wf.mean_train_ic != 0:
            score = max(0.0, min(1.0, wf.ic_degradation / abs(wf.mean_train_ic)))
        else:
            score = 0.0

        # OOS significance test
        oos_series = pd.Series(wf.test_ics)
        t_stat = float(oos_series.mean() / oos_series.std(ddof=1) * np.sqrt(len(oos_series)))
        p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=len(oos_series) - 1)))
        is_sig = p_val < self.alpha and wf.mean_test_ic > 0

        # Recommendation
        if score < 0.2:
            rec = "PASS: Low overfitting (OOS IC is >80% of IS IC). Signal appears robust."
        elif score < 0.5:
            rec = "CAUTION: Moderate degradation (OOS IC is 50-80% of IS IC). Monitor in live."
        elif score < 0.8:
            rec = "WARNING: Significant degradation (OOS IC < 50% of IS IC). Consider regularisation."
        else:
            rec = "FAIL: Severe overfitting. OOS IC is near zero or negative."

        if wf.mean_test_ic <= 0:
            rec = "FAIL: OOS IC is non-positive. Signal does not generalise."

        return OverfittingDiagnosis(
            n_splits=wf.n_splits,
            mean_is_ic=wf.mean_train_ic,
            mean_oos_ic=wf.mean_test_ic,
            ic_degradation=wf.ic_degradation,
            overfitting_score=score,
            is_significant_oos=is_sig,
            recommendation=rec,
        )

    # ------------------------------------------------------------------ #
    # Multiple testing correction
    # ------------------------------------------------------------------ #

    def multiple_testing_correction(
        self,
        p_values: Dict[str, float],
        alpha: Optional[float] = None,
        method: str = "bh",
    ) -> MultipleTestingResult:
        """Apply multiple-testing corrections to a set of p-values.

        Methods:
          - 'bonferroni': conservative family-wise error rate control
          - 'bh': Benjamini-Hochberg FDR control
          - 'both': compute both

        Parameters
        ----------
        p_values : dict of hypothesis -> raw p-value
        alpha    : significance level (default self.alpha)
        method   : 'bonferroni', 'bh', or 'both'

        Returns
        -------
        MultipleTestingResult
        """
        alpha_ = alpha or self.alpha
        names = list(p_values.keys())
        raw_p = np.array([p_values[n] for n in names])
        n = len(raw_p)

        # Bonferroni
        bonf_p = dict(zip(names, np.minimum(raw_p * n, 1.0).tolist()))
        bonf_sig = {k: v < alpha_ for k, v in bonf_p.items()}

        # Benjamini-Hochberg
        sorted_idx = np.argsort(raw_p)
        sorted_p = raw_p[sorted_idx]
        bh_thresholds = np.arange(1, n + 1) / n * alpha_

        # BH correction: find largest k such that p_(k) <= k*alpha/n
        bh_p_adj = np.ones(n)
        for k in range(n - 1, -1, -1):
            if k == n - 1:
                bh_p_adj[k] = sorted_p[k]
            else:
                bh_p_adj[k] = min(sorted_p[k] * n / (k + 1), bh_p_adj[k + 1])

        # Unsort
        bh_p_unsorted = np.empty(n)
        bh_p_unsorted[sorted_idx] = bh_p_adj
        bh_p_dict = dict(zip(names, bh_p_unsorted.tolist()))
        bh_sig = {k: v < alpha_ for k, v in bh_p_dict.items()}

        return MultipleTestingResult(
            raw_p_values=p_values,
            bonferroni_p=bonf_p,
            bh_p=bh_p_dict,
            significant_bonferroni=bonf_sig,
            significant_bh=bh_sig,
            alpha=alpha_,
            n_tests=n,
        )

    # ------------------------------------------------------------------ #
    # Sharpe ratio significance
    # ------------------------------------------------------------------ #

    def sharpe_significance_test(
        self,
        returns: pd.Series,
        target_sharpe: float = 0.0,
        bars_per_year: int = 252,
        method: str = "lo",
    ) -> Tuple[float, float, float]:
        """Test whether annualised Sharpe is significantly above target.

        Uses the Lo (2002) asymptotic t-test correcting for autocorrelation.

        Parameters
        ----------
        returns      : return series
        target_sharpe: null hypothesis Sharpe (default 0)
        bars_per_year: periods per year
        method       : 'lo' (Lo 2002) or 'simple' (iid assumption)

        Returns
        -------
        (sharpe_estimate, t_stat, p_value) — one-tailed
        """
        from research.signal_analytics.utils import sharpe_ratio

        r = returns.dropna().values
        n = len(r)
        if n < 10:
            return float("nan"), float("nan"), float("nan")

        sr = sharpe_ratio(r, bars_per_year=bars_per_year)

        if method == "lo":
            # Lo (2002): adjust standard error for autocorrelation
            mu = np.mean(r)
            sig = np.std(r, ddof=1)
            sr_raw = mu / sig  # per-period Sharpe

            # Compute lag-1 autocorrelation of r
            rho1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])
            # Asymptotic variance of SR under IID + autocorrelation adjustment
            adjustment = 1 + (2 * rho1 / (1 - rho1)) if abs(rho1) < 0.99 else 1.0
            var_sr = (1 + 0.5 * sr_raw**2) * adjustment / n
            se_sr_annual = np.sqrt(var_sr * bars_per_year)
            t_stat = (sr - target_sharpe) / se_sr_annual if se_sr_annual > 0 else float("nan")
        else:
            # Simple: SE of SR is sqrt((1 + SR^2/2) / n)
            se_sr = np.sqrt((1 + (sr / np.sqrt(bars_per_year))**2 / 2) / n)
            t_stat = (sr - target_sharpe) / se_sr if se_sr > 0 else float("nan")

        p_val = float(1 - stats.t.cdf(float(t_stat), df=n - 1)) if not np.isnan(t_stat) else float("nan")
        return sr, float(t_stat), p_val

    # ------------------------------------------------------------------ #
    # Lead-lag analysis
    # ------------------------------------------------------------------ #

    def lead_lag_analysis(
        self,
        signal: pd.Series,
        returns: pd.Series,
        max_lag: int = 10,
        method: str = "spearman",
    ) -> pd.DataFrame:
        """Cross-correlation of signal with lagged/led returns.

        Answers: does the signal lead returns (predictive) or lag returns
        (reactive)?

        Parameters
        ----------
        signal  : signal time-series
        returns : return time-series
        max_lag : maximum lag/lead to test (both directions)
        method  : correlation method

        Returns
        -------
        pd.DataFrame[lag -> (correlation, p_value)]
          Negative lag = signal lags returns (reactive)
          Positive lag = signal leads returns (predictive)
        """
        df = pd.concat({"sig": signal, "ret": returns}, axis=1).dropna()
        records: list[dict] = []

        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # Signal lags returns: compare sig[lag:] with ret[:lag]
                s = df["sig"].iloc[-lag:].values
                r = df["ret"].iloc[:lag].values
            elif lag > 0:
                # Signal leads returns: compare sig[:-lag] with ret[lag:]
                s = df["sig"].iloc[:-lag].values
                r = df["ret"].iloc[lag:].values
            else:
                s = df["sig"].values
                r = df["ret"].values

            if len(s) < 3:
                records.append({"lag": lag, "correlation": float("nan"), "p_value": float("nan")})
                continue

            if method == "spearman":
                corr, p = stats.spearmanr(s, r)
            else:
                corr, p = stats.pearsonr(s, r)
            records.append({"lag": lag, "correlation": float(corr), "p_value": float(p)})

        result = pd.DataFrame(records).set_index("lag")
        result["significant"] = result["p_value"] < self.alpha
        return result

    # ------------------------------------------------------------------ #
    # Autocorrelation test (Ljung-Box)
    # ------------------------------------------------------------------ #

    def ljung_box_test(
        self,
        series: pd.Series,
        lags: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Ljung-Box test for autocorrelation in a time-series.

        Tests H0: no autocorrelation up to lag k.

        Parameters
        ----------
        series : time-series (e.g. IC series or residuals)
        lags   : list of lags to test (default [5, 10, 15, 20])

        Returns
        -------
        pd.DataFrame[lag x (lb_stat, p_value, significant)]
        """
        if lags is None:
            lags = [5, 10, 15, 20]

        clean = series.dropna().values
        n = len(clean)
        if n < max(lags) + 2:
            return pd.DataFrame()

        # Compute autocorrelations
        mean_c = clean.mean()
        var_c = np.var(clean, ddof=1)

        records: list[dict] = []
        for max_lag in lags:
            acf_sum = 0.0
            for k in range(1, max_lag + 1):
                acf_k = np.mean((clean[:n - k] - mean_c) * (clean[k:] - mean_c)) / var_c if var_c > 0 else 0.0
                acf_sum += acf_k**2 / (n - k)

            lb_stat = n * (n + 2) * acf_sum
            p_val = float(1 - stats.chi2.cdf(lb_stat, df=max_lag))
            records.append({
                "lag": max_lag,
                "lb_stat": float(lb_stat),
                "p_value": p_val,
                "significant": p_val < self.alpha,
            })

        return pd.DataFrame(records).set_index("lag")

    # ------------------------------------------------------------------ #
    # Signal consistency (regime-based)
    # ------------------------------------------------------------------ #

    def regime_consistency_test(
        self,
        trades: pd.DataFrame,
        signal_col: str,
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
        regime_col: str = "regime",
    ) -> pd.DataFrame:
        """Test whether IC is consistent across regimes.

        Parameters
        ----------
        trades        : trade records
        signal_col    : signal column
        return_col    : P&L column
        dollar_pos_col: position column
        regime_col    : regime label column

        Returns
        -------
        pd.DataFrame[regime x (ic, t_stat, p_value, n_obs, significant)]
        """
        if regime_col not in trades.columns:
            raise ValueError(f"trades must have '{regime_col}' column")

        df = trades.copy()
        if dollar_pos_col in df.columns:
            pos = df[dollar_pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[return_col] / pos
        else:
            df["_ret"] = df[return_col]

        records: list[dict] = []
        for regime in df[regime_col].unique():
            sub = df[df[regime_col] == regime][[signal_col, "_ret"]].dropna()
            n = len(sub)
            if n < 3:
                records.append({"regime": str(regime), "ic": float("nan"), "t_stat": float("nan"),
                                 "p_value": float("nan"), "n_obs": n, "significant": False})
                continue
            r, p = stats.spearmanr(sub[signal_col], sub["_ret"])
            t = float(r) * np.sqrt(n - 2) / max(np.sqrt(max(1 - float(r)**2, 1e-10)), 1e-10)
            records.append({
                "regime": str(regime),
                "ic": float(r),
                "t_stat": float(t),
                "p_value": float(p),
                "n_obs": n,
                "significant": float(p) < self.alpha,
            })

        return pd.DataFrame(records).set_index("regime")

    # ------------------------------------------------------------------ #
    # IC stability across symbols
    # ------------------------------------------------------------------ #

    def symbol_ic_stability_test(
        self,
        trades: pd.DataFrame,
        signal_col: str,
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
        min_obs: int = 10,
    ) -> Dict[str, float]:
        """Test IC stability across symbols using Cochran's Q test.

        Cochran's Q tests whether ICs are homogeneous across groups.
        H0: all ICs are equal.

        Parameters
        ----------
        trades        : trade records with 'sym' column
        signal_col    : signal column
        return_col    : P&L column
        dollar_pos_col: position column
        min_obs       : minimum obs per symbol

        Returns
        -------
        Dict with keys: cochran_q, p_value, is_heterogeneous, n_symbols,
                        mean_ic, std_ic, cv_ic
        """
        if "sym" not in trades.columns:
            return {}

        df = trades.copy()
        if dollar_pos_col in df.columns:
            pos = df[dollar_pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[return_col] / pos
        else:
            df["_ret"] = df[return_col]

        sym_ics: list[float] = []
        sym_ns: list[int] = []

        for sym in df["sym"].unique():
            sub = df[df["sym"] == sym][[signal_col, "_ret"]].dropna()
            if len(sub) < min_obs:
                continue
            r, _ = stats.spearmanr(sub[signal_col], sub["_ret"])
            sym_ics.append(float(r))
            sym_ns.append(len(sub))

        if len(sym_ics) < 2:
            return {}

        ic_arr = np.array(sym_ics)
        n_arr = np.array(sym_ns)
        k = len(ic_arr)
        n_total = n_arr.sum()

        # Cochran's Q statistic for proportions (adapted for correlations)
        # Simple chi-squared test: each IC vs mean IC, weighted by n
        mean_ic = np.average(ic_arr, weights=n_arr)
        Q = float(np.sum(n_arr * (ic_arr - mean_ic)**2))
        p_val = float(1 - stats.chi2.cdf(Q, df=k - 1))

        return {
            "cochran_q": Q,
            "p_value": p_val,
            "is_heterogeneous": p_val < self.alpha,
            "n_symbols": k,
            "mean_ic": float(mean_ic),
            "std_ic": float(np.std(ic_arr, ddof=1)),
            "cv_ic": float(np.std(ic_arr, ddof=1) / abs(mean_ic)) if mean_ic != 0 else float("nan"),
        }

    # ------------------------------------------------------------------ #
    # Calmar-Sharpe trade-off
    # ------------------------------------------------------------------ #

    def return_risk_profile(
        self,
        trades: pd.DataFrame,
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
        bars_per_year: int = 252,
    ) -> Dict[str, float]:
        """Compute comprehensive return/risk metrics.

        Parameters
        ----------
        trades        : trade records
        return_col    : P&L column
        dollar_pos_col: position column
        bars_per_year : periods per year

        Returns
        -------
        Dict with sharpe, sortino, calmar, max_dd, skew, kurtosis
        """
        from research.signal_analytics.utils import performance_summary

        df = trades.copy()
        if dollar_pos_col in df.columns:
            pos = df[dollar_pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[return_col] / pos
        else:
            df["_ret"] = df[return_col]

        return performance_summary(df["_ret"].dropna(), bars_per_year=bars_per_year)

    # ------------------------------------------------------------------ #
    # Correlation of signal to known risk factors
    # ------------------------------------------------------------------ #

    def market_factor_correlation(
        self,
        signal: pd.Series,
        market_returns: pd.Series,
        method: str = "spearman",
    ) -> Dict[str, float]:
        """Measure correlation of signal to market-wide returns.

        High correlation suggests the signal is beta-driven rather than
        genuine alpha.

        Parameters
        ----------
        signal         : signal values indexed by time/trade
        market_returns : market return series (same index)
        method         : correlation method

        Returns
        -------
        Dict with corr, p_value, is_market_correlated
        """
        df = pd.concat({"sig": signal, "mkt": market_returns}, axis=1).dropna()
        if len(df) < 3:
            return {}

        if method == "spearman":
            r, p = stats.spearmanr(df["sig"], df["mkt"])
        else:
            r, p = stats.pearsonr(df["sig"], df["mkt"])

        return {
            "correlation": float(r),
            "p_value": float(p),
            "is_market_correlated": abs(r) > 0.3 and float(p) < self.alpha,
        }

    # ------------------------------------------------------------------ #
    # Visualisations
    # ------------------------------------------------------------------ #

    def plot_permutation_test(
        self,
        perm_result: PermutationTestResult,
        save_path: Optional[str | Path] = None,
        title: str = "IC Permutation Test",
    ) -> plt.Figure:
        """Plot null distribution with observed IC highlighted.

        Parameters
        ----------
        perm_result : PermutationTestResult from permutation_ic_test()
        save_path   : optional save path
        title       : figure title
        """
        fig, ax = plt.subplots(figsize=(9, 5))
        null = perm_result.null_distribution

        ax.hist(null, bins=50, density=True, color="#bdc3c7", alpha=0.8, label="Null distribution")
        ax.axvline(perm_result.observed_ic, color="#e74c3c", linewidth=2,
                   label=f"Observed IC={perm_result.observed_ic:.4f}")
        ax.axvline(perm_result.ci_upper, color="#3498db", linewidth=1.2, linestyle="--",
                   label=f"{perm_result.confidence_level:.0%} CI: [{perm_result.ci_lower:.3f}, {perm_result.ci_upper:.3f}]")
        ax.axvline(perm_result.ci_lower, color="#3498db", linewidth=1.2, linestyle="--")
        ax.axvline(0, color="black", linewidth=0.8)

        sig_str = "SIGNIFICANT" if perm_result.is_significant else "NOT significant"
        ax.set_title(f"{title}\np={perm_result.p_value:.4f}  ({sig_str})")
        ax.set_xlabel("IC")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_walk_forward_ic(
        self,
        wf_result: WalkForwardResult,
        save_path: Optional[str | Path] = None,
        title: str = "Walk-Forward IC Validation",
    ) -> plt.Figure:
        """Bar chart comparing in-sample vs out-of-sample IC per fold.

        Parameters
        ----------
        wf_result : WalkForwardResult from walk_forward_ic()
        save_path : optional save path
        title     : figure title
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        n = len(wf_result.train_ics)
        x = np.arange(n)
        width = 0.35

        ax.bar(x - width/2, wf_result.train_ics, width, label="In-sample (train)",
               color="#3498db", alpha=0.8)
        ax.bar(x + width/2, wf_result.test_ics, width, label="Out-of-sample (test)",
               color="#e74c3c", alpha=0.8)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(wf_result.mean_train_ic, color="#3498db", linewidth=1.2, linestyle="--", alpha=0.7,
                   label=f"Mean IS={wf_result.mean_train_ic:.4f}")
        ax.axhline(wf_result.mean_test_ic, color="#e74c3c", linewidth=1.2, linestyle="--", alpha=0.7,
                   label=f"Mean OOS={wf_result.mean_test_ic:.4f}")

        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {i+1}" for i in range(n)])
        ax.set_ylabel("IC")
        ax.set_title(
            f"{title}\nIC Degradation={wf_result.ic_degradation:.4f}  "
            f"OOS/IS Ratio={wf_result.overfitting_ratio:.2f}"
        )
        ax.legend()
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_lead_lag(
        self,
        lead_lag_df: pd.DataFrame,
        save_path: Optional[str | Path] = None,
        title: str = "Lead-Lag Analysis",
    ) -> plt.Figure:
        """Plot lead-lag correlation across lags.

        Parameters
        ----------
        lead_lag_df : output of lead_lag_analysis()
        save_path   : optional save path
        title       : figure title
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        lags = lead_lag_df.index
        corr = lead_lag_df["correlation"].values
        sig = lead_lag_df["significant"].values

        colors = ["#e74c3c" if s else "#bdc3c7" for s in sig]
        bars = ax.bar(lags, corr, color=colors, alpha=0.8)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--")

        # Annotate significant bars
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="#e74c3c", alpha=0.8, label=f"Significant (p<{self.alpha})"),
            Patch(facecolor="#bdc3c7", alpha=0.8, label="Not significant"),
        ])

        ax.set_xlabel("Lag (negative = signal lags, positive = signal leads)")
        ax.set_ylabel("Correlation")
        ax.set_title(title)
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_regime_ic_comparison(
        self,
        regime_ic_df: pd.DataFrame,
        save_path: Optional[str | Path] = None,
        title: str = "IC by Regime",
    ) -> plt.Figure:
        """Bar chart of IC per regime with significance stars.

        Parameters
        ----------
        regime_ic_df : output of regime_consistency_test()
        save_path    : optional save path
        title        : figure title
        """
        fig, ax = plt.subplots(figsize=(9, 5))
        regimes = regime_ic_df.index.tolist()
        ics = regime_ic_df["ic"].values
        sig = regime_ic_df["significant"].values if "significant" in regime_ic_df.columns else [False] * len(ics)

        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in ics]
        bars = ax.bar(regimes, ics, color=colors, alpha=0.8, edgecolor="white")

        for bar, s, n_obs_val in zip(bars, sig, regime_ic_df.get("n_obs", pd.Series([0]*len(regimes))).values):
            if s:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002 * np.sign(bar.get_height()),
                    "*", ha="center", va="bottom", fontsize=14, color="black",
                )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Regime")
        ax.set_ylabel("IC")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    # ------------------------------------------------------------------ #
    # Comprehensive diagnostic report
    # ------------------------------------------------------------------ #

    def full_diagnostic_report(
        self,
        trades: pd.DataFrame,
        signal_col: str,
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
        output_dir: Optional[str | Path] = None,
        n_permutations: int = 500,
        n_wf_splits: int = 5,
    ) -> Dict[str, object]:
        """Run all diagnostic tests and optionally save plots.

        Parameters
        ----------
        trades          : trade records
        signal_col      : signal column
        return_col      : P&L column
        dollar_pos_col  : position column
        output_dir      : directory to save plots (optional)
        n_permutations  : permutation test iterations
        n_wf_splits     : walk-forward splits

        Returns
        -------
        Dict of diagnostic results
        """
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
        else:
            out_path = None

        df = trades.copy()
        if dollar_pos_col in df.columns:
            pos = df[dollar_pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[return_col] / pos
        else:
            df["_ret"] = df[return_col]

        result: dict = {}

        # Permutation test
        sub = df[[signal_col, "_ret"]].dropna()
        if len(sub) >= 5:
            perm = self.permutation_ic_test(sub[signal_col], sub["_ret"], n_permutations=n_permutations)
            result["permutation_test"] = perm
            if out_path:
                fig = self.plot_permutation_test(perm, save_path=out_path / "permutation_test.png")
                plt.close(fig)

        # Walk-forward
        wf = self.walk_forward_ic(trades, signal_col, return_col, dollar_pos_col, n_splits=n_wf_splits)
        result["walk_forward"] = wf
        if out_path and wf.n_splits > 0:
            fig = self.plot_walk_forward_ic(wf, save_path=out_path / "walk_forward_ic.png")
            plt.close(fig)

        # Overfitting
        result["overfitting"] = self.overfitting_diagnosis(trades, signal_col, return_col, dollar_pos_col, n_wf_splits)

        # Regime consistency
        if "regime" in trades.columns:
            reg_ic = self.regime_consistency_test(trades, signal_col, return_col, dollar_pos_col)
            result["regime_consistency"] = reg_ic
            if out_path:
                fig = self.plot_regime_ic_comparison(reg_ic, save_path=out_path / "regime_ic.png")
                plt.close(fig)

        # Symbol consistency
        if "sym" in trades.columns:
            result["symbol_consistency"] = self.symbol_ic_stability_test(trades, signal_col, return_col, dollar_pos_col)

        # Return/risk profile
        result["risk_profile"] = self.return_risk_profile(trades, return_col, dollar_pos_col)

        # Ljung-Box on residuals
        if len(sub) > 30:
            residuals = sub["_ret"] - sub[signal_col].rank(pct=True) * sub["_ret"].std(ddof=1)
            lb_result = self.ljung_box_test(residuals)
            result["ljung_box"] = lb_result

        return result

    def ic_variance_inflation_test(
        self,
        trades: "pd.DataFrame",
        signal_cols: list[str],
        return_col: str = "_ret",
    ) -> "pd.DataFrame":
        """
        Test whether high pairwise signal correlation inflates apparent IC
        by computing partial IC (controlling for correlated signals).

        Partial IC of signal_k controlling for all other signals is
        estimated via residual OLS: regress signal_k on others, take residuals,
        then correlate with return.

        Parameters
        ----------
        trades : pd.DataFrame
            Trade log with signal and return columns.
        signal_cols : list of str
            List of signal columns to test.
        return_col : str
            Return column.

        Returns
        -------
        pd.DataFrame with columns [signal, raw_ic, partial_ic,
                                    ic_inflation_pct, vif].
        """
        from scipy import stats as _stats

        df = trades[signal_cols + [return_col]].dropna()
        if len(df) < len(signal_cols) + 5:
            raise ValueError("Insufficient observations.")

        ret = df[return_col].values.astype(float)
        rows: list[dict] = []

        for k, sname in enumerate(signal_cols):
            sig = df[sname].values.astype(float)
            # Raw IC
            raw_rho, _ = _stats.spearmanr(sig, ret)

            # Partial IC: residualise signal_k against all other signals
            others = [c for c in signal_cols if c != sname]
            if others:
                X_oth = df[others].values.astype(float)
                X_c = np.column_stack([np.ones(len(X_oth)), X_oth])
                try:
                    coef, _, _, _ = np.linalg.lstsq(X_c, sig, rcond=None)
                    sig_resid = sig - X_c @ coef
                except np.linalg.LinAlgError:
                    sig_resid = sig
                partial_rho, _ = _stats.spearmanr(sig_resid, ret)
                # VIF: 1 / (1 - R^2) where R^2 is from regressing signal_k on others
                fitted = X_c @ coef
                ss_res = float(np.sum((sig - fitted) ** 2))
                ss_tot = float(np.sum((sig - sig.mean()) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
            else:
                partial_rho = raw_rho
                vif = 1.0

            inflation_pct = (abs(raw_rho) - abs(partial_rho)) / abs(raw_rho) * 100 if raw_rho != 0 else 0.0
            rows.append({
                "signal": sname,
                "raw_ic": float(raw_rho),
                "partial_ic": float(partial_rho),
                "ic_inflation_pct": float(inflation_pct),
                "vif": float(vif),
            })

        return pd.DataFrame(rows)

    def ic_stability_score(
        self,
        ic_series: "pd.Series",
        window: int = 20,
    ) -> dict:
        """
        Compute a composite IC stability score (0-100) from the rolling IC
        time series.

        Components
        ----------
        - Proportion of windows with positive IC (weight 40%)
        - 1 - CV of rolling IC (weight 30%), where CV = std/abs(mean)
        - Proportion of windows where |IC| > 0.02 (weight 30%)

        Parameters
        ----------
        ic_series : pd.Series
            Time series of IC values (one per rebalancing period).
        window : int
            Sub-window for stability evaluation.

        Returns
        -------
        dict with keys: stability_score, pct_positive, inv_cv, pct_meaningful,
                        ic_mean, ic_std.
        """
        arr = np.asarray(ic_series.dropna(), dtype=float)
        if len(arr) < 4:
            return {"stability_score": np.nan}

        ic_mean = float(np.mean(arr))
        ic_std = float(np.std(arr, ddof=1))
        pct_pos = float((arr > 0).mean())
        pct_meaningful = float((np.abs(arr) > 0.02).mean())
        cv = ic_std / abs(ic_mean) if ic_mean != 0 else np.inf
        inv_cv = max(0.0, 1.0 - min(cv, 2.0) / 2.0)

        score = 100.0 * (0.4 * pct_pos + 0.3 * inv_cv + 0.3 * pct_meaningful)

        return {
            "stability_score": round(score, 2),
            "pct_positive": pct_pos,
            "inv_cv": inv_cv,
            "pct_meaningful": pct_meaningful,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "n_periods": len(arr),
        }
