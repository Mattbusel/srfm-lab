"""
statistical_tests.py -- comprehensive statistical testing for trading signals and models.

Covers normality, stationarity, signal significance, and model diagnostics.
All tests return TestResult dataclasses for uniform downstream consumption.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2, f as f_dist


# ---------------------------------------------------------------------------
# Core result types
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """Uniform container for statistical test output."""

    statistic: float
    p_value: float
    is_significant: bool = field(init=False)
    interpretation: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.is_significant = self.p_value < 0.05

    def __repr__(self) -> str:
        sig = "significant" if self.is_significant else "not significant"
        return (
            f"TestResult(stat={self.statistic:.4f}, p={self.p_value:.4f}, {sig})"
        )


# ---------------------------------------------------------------------------
# Normality Tests
# ---------------------------------------------------------------------------

class NormalityTests:
    """
    Collection of normality tests for return series and residuals.

    All methods accept a 1-D numpy array and return a TestResult. NaN values
    are silently dropped before testing.
    """

    @staticmethod
    def _clean(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < 8:
            raise ValueError(f"Need at least 8 finite observations; got {len(x)}")
        return x

    @classmethod
    def shapiro_wilk(cls, x: np.ndarray) -> TestResult:
        """
        Shapiro-Wilk normality test.

        Null hypothesis: sample is drawn from a normal distribution.
        Best suited to samples of size <= 5000.
        """
        x = cls._clean(x)
        if len(x) > 5000:
            # Use only a random subsample to stay in the valid range
            rng = np.random.default_rng(seed=42)
            x = rng.choice(x, size=5000, replace=False)
        stat, p = stats.shapiro(x)
        interp = (
            "Reject normality (p < 0.05)" if p < 0.05
            else "Cannot reject normality (p >= 0.05)"
        )
        return TestResult(statistic=float(stat), p_value=float(p), interpretation=interp)

    @classmethod
    def jarque_bera(cls, x: np.ndarray) -> TestResult:
        """
        Jarque-Bera test.

        Null hypothesis: sample skewness and excess kurtosis are jointly zero
        (i.e., data are normally distributed). The test statistic is asymptotically
        chi-squared with 2 degrees of freedom.
        """
        x = cls._clean(x)
        n = len(x)
        s = stats.skew(x)
        k = stats.kurtosis(x)  # excess kurtosis
        jb_stat = (n / 6.0) * (s**2 + (k**2) / 4.0)
        p = float(1.0 - chi2.cdf(jb_stat, df=2))
        interp = (
            "Non-normal: significant skewness/kurtosis" if p < 0.05
            else "Cannot reject normality via JB test"
        )
        return TestResult(
            statistic=float(jb_stat),
            p_value=p,
            interpretation=interp,
            extra={"skewness": float(s), "excess_kurtosis": float(k)},
        )

    @classmethod
    def kolmogorov_smirnov(cls, x: np.ndarray, dist: str = "norm") -> TestResult:
        """
        Kolmogorov-Smirnov test against a reference distribution.

        Parameters
        ----------
        x    : sample data
        dist : scipy.stats distribution name (default 'norm')

        The test fits the distribution parameters from the data before testing,
        which is Lilliefors-like -- the p-value is therefore approximate.
        """
        x = cls._clean(x)
        dist_obj = getattr(stats, dist)
        params = dist_obj.fit(x)
        stat, p = stats.kstest(x, dist, args=params)
        interp = (
            f"Reject {dist} distribution (p < 0.05)" if p < 0.05
            else f"Cannot reject {dist} distribution (p >= 0.05)"
        )
        return TestResult(
            statistic=float(stat),
            p_value=float(p),
            interpretation=interp,
            extra={"fitted_params": params, "distribution": dist},
        )

    @classmethod
    def anderson_darling(cls, x: np.ndarray) -> TestResult:
        """
        Anderson-Darling test for normality.

        scipy.stats.anderson returns critical values at significance levels
        [15%, 10%, 5%, 2.5%, 1%]. We report whether the statistic exceeds the
        5% critical value and compute an approximate p-value via interpolation.
        """
        x = cls._clean(x)
        result = stats.anderson(x, dist="norm")
        stat = float(result.statistic)

        # significance levels returned by scipy: [15, 10, 5, 2.5, 1] percent
        sig_levels = np.array([0.15, 0.10, 0.05, 0.025, 0.01])
        crit_values = result.critical_values

        # Interpolate approximate p-value
        if stat <= crit_values[0]:
            p_approx = 0.15
        elif stat >= crit_values[-1]:
            p_approx = 0.01
        else:
            p_approx = float(np.interp(stat, crit_values[::-1], sig_levels[::-1]))

        sig_at_5pct = stat > crit_values[2]  # index 2 => 5% level
        interp = (
            "Reject normality at 5% (Anderson-Darling)" if sig_at_5pct
            else "Cannot reject normality (Anderson-Darling)"
        )
        return TestResult(
            statistic=stat,
            p_value=p_approx,
            interpretation=interp,
            extra={
                "critical_values": dict(zip(["15%", "10%", "5%", "2.5%", "1%"], crit_values))
            },
        )


# ---------------------------------------------------------------------------
# Stationarity Tests
# ---------------------------------------------------------------------------

class StationarityTests:
    """
    Unit root and stationarity tests for time series data.

    ADF and KPSS have opposite null hypotheses:
      ADF  -- H0: unit root present (non-stationary)
      KPSS -- H0: series is stationary
    """

    @staticmethod
    def _clean(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < 20:
            raise ValueError(f"Need at least 20 observations for stationarity tests; got {len(x)}")
        return x

    @classmethod
    def augmented_dickey_fuller(
        cls, x: np.ndarray, lags: Union[int, str] = "auto"
    ) -> TestResult:
        """
        Augmented Dickey-Fuller test for a unit root.

        Parameters
        ----------
        x    : time series
        lags : 'auto' selects via AIC; otherwise integer lag count

        Returns TestResult where is_significant == True means the series is
        stationary (we reject the null of a unit root).
        """
        from statsmodels.tsa.stattools import adfuller

        x = cls._clean(x)
        autolag = "AIC" if lags == "auto" else None
        maxlags = None if lags == "auto" else int(lags)

        try:
            result = adfuller(x, maxlag=maxlags, autolag=autolag, regression="c")
        except Exception as exc:
            raise RuntimeError(f"ADF test failed: {exc}") from exc

        adf_stat = float(result[0])
        p_val = float(result[1])
        used_lags = int(result[2])
        n_obs = int(result[3])
        crit = result[4]  # dict: {'1%': ..., '5%': ..., '10%': ...}

        is_stationary = p_val < 0.05
        interp = (
            f"Stationary at 5% (ADF stat={adf_stat:.4f}, p={p_val:.4f})"
            if is_stationary
            else f"Non-stationary / unit root (ADF stat={adf_stat:.4f}, p={p_val:.4f})"
        )
        return TestResult(
            statistic=adf_stat,
            p_value=p_val,
            interpretation=interp,
            extra={
                "lags_used": used_lags,
                "n_obs": n_obs,
                "critical_values": crit,
                "is_stationary": is_stationary,
            },
        )

    @classmethod
    def kpss(cls, x: np.ndarray) -> TestResult:
        """
        KPSS test. Null hypothesis: series is stationary around a constant.

        is_significant == True means we REJECT stationarity (non-stationary).
        """
        from statsmodels.tsa.stattools import kpss as sm_kpss

        x = cls._clean(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p_val, lags, crit = sm_kpss(x, regression="c", nlags="auto")

        is_non_stationary = p_val < 0.05
        interp = (
            "Reject stationarity (KPSS)" if is_non_stationary
            else "Cannot reject stationarity (KPSS)"
        )
        return TestResult(
            statistic=float(stat),
            p_value=float(p_val),
            interpretation=interp,
            extra={"lags": lags, "critical_values": crit},
        )

    @classmethod
    def zivot_andrews(cls, x: np.ndarray) -> TestResult:
        """
        Zivot-Andrews test -- unit root test allowing for a single structural break.

        Null: unit root with no structural break.
        Rejection implies the series is stationary with one break point.
        """
        from statsmodels.tsa.stattools import zivot_andrews as sm_za

        x = cls._clean(x)
        try:
            result = sm_za(x, maxlag=None, regression="c", autolag="AIC")
        except Exception as exc:
            raise RuntimeError(f"Zivot-Andrews test failed: {exc}") from exc

        stat = float(result[0])
        p_val = float(result[1])
        # statsmodels ZA p-value is from a tabulated distribution
        interp = (
            "Reject unit root with structural break (stationary)" if p_val < 0.05
            else "Cannot reject unit root (non-stationary or break present)"
        )
        return TestResult(
            statistic=stat,
            p_value=p_val,
            interpretation=interp,
            extra={"breakpoint_index": int(result[4]) if len(result) > 4 else None},
        )

    @classmethod
    def variance_ratio_test(
        cls, x: np.ndarray, periods: list[int] = None
    ) -> dict:
        """
        Lo-MacKinlay variance ratio test for the random walk hypothesis.

        For each holding period q, the variance ratio VR(q) = Var(q-period return)
        / (q * Var(1-period return)). Under the random walk, VR(q) = 1.

        Returns a dict keyed by period with TestResult values.
        """
        if periods is None:
            periods = [2, 4, 8, 16]

        x = cls._clean(x)
        n = len(x)
        # compute 1-period variance
        mu = np.mean(np.diff(x)) if len(x) > 1 else 0.0
        # treat x as log prices; compute returns
        returns = np.diff(x)
        n_ret = len(returns)
        var_1 = float(np.var(returns, ddof=1))

        results: dict = {}
        for q in periods:
            if q >= n_ret:
                results[q] = TestResult(
                    statistic=np.nan,
                    p_value=np.nan,
                    interpretation=f"Insufficient data for q={q}",
                )
                continue

            # overlapping q-period returns
            q_returns = np.array([np.sum(returns[i: i + q]) for i in range(n_ret - q + 1)])
            var_q = float(np.var(q_returns, ddof=1)) / q

            vr = var_q / var_1 if var_1 > 0 else np.nan

            # heteroscedasticity-robust z-statistic (Lo-MacKinlay 1988)
            # simplified delta computation
            m = q * (n_ret - q + 1) * (1.0 - q / n_ret)
            delta = np.zeros(q - 1)
            for j in range(1, q):
                numer = np.sum(
                    (returns[j:] - mu) ** 2 * (returns[: n_ret - j] - mu) ** 2
                )
                denom = (np.sum((returns - mu) ** 2)) ** 2
                delta[j - 1] = numer / denom * n_ret**2

            theta = float(np.sum(((2 * (q - np.arange(1, q))) / q) ** 2 * delta))
            z_stat = (vr - 1.0) / np.sqrt(theta) if theta > 0 else np.nan
            p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))) if np.isfinite(z_stat) else np.nan

            interp = (
                f"Reject random walk at q={q} (VR={vr:.4f})" if (np.isfinite(p_val) and p_val < 0.05)
                else f"Cannot reject random walk at q={q} (VR={vr:.4f})"
            )
            results[q] = TestResult(
                statistic=float(z_stat) if np.isfinite(z_stat) else np.nan,
                p_value=p_val if np.isfinite(p_val) else np.nan,
                interpretation=interp,
                extra={"variance_ratio": vr, "q": q},
            )
        return results


# ---------------------------------------------------------------------------
# Signal Tests
# ---------------------------------------------------------------------------

class SignalTests:
    """
    Tests for trading signal quality and predictive power.
    """

    @staticmethod
    def t_test_mean_zero(returns: np.ndarray) -> TestResult:
        """
        One-sample t-test: is the mean return significantly different from zero?

        Null hypothesis: mean == 0 (no edge).
        """
        x = np.asarray(returns, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < 4:
            raise ValueError("Need at least 4 observations")
        stat, p = stats.ttest_1samp(x, popmean=0.0)
        interp = (
            "Mean return significantly != 0 (edge detected)" if p < 0.05
            else "Mean return not significantly different from 0"
        )
        return TestResult(
            statistic=float(stat),
            p_value=float(p),
            interpretation=interp,
            extra={"mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)), "n": len(x)},
        )

    @staticmethod
    def spearman_ic_test(signals: np.ndarray, returns: np.ndarray) -> TestResult:
        """
        Information Coefficient (IC) test using Spearman rank correlation.

        Tests whether the IC is significantly different from zero under the
        null that signals and returns are uncorrelated.
        """
        s = np.asarray(signals, dtype=float)
        r = np.asarray(returns, dtype=float)
        mask = np.isfinite(s) & np.isfinite(r)
        s, r = s[mask], r[mask]
        if len(s) < 10:
            raise ValueError("Need at least 10 paired observations for IC test")
        ic, p = stats.spearmanr(s, r)
        interp = (
            f"IC={ic:.4f} is significant (signal has predictive value)" if p < 0.05
            else f"IC={ic:.4f} not significant (signal may be noise)"
        )
        return TestResult(
            statistic=float(ic),
            p_value=float(p),
            interpretation=interp,
            extra={"n": int(len(s))},
        )

    @staticmethod
    def granger_causality(
        x: np.ndarray, y: np.ndarray, max_lag: int = 5
    ) -> dict:
        """
        Granger causality: does x Granger-cause y?

        For each lag from 1 to max_lag, fits a restricted model (y on lags of y)
        and an unrestricted model (y on lags of y and x), then computes an F-test.

        Returns dict with keys 1..max_lag, each a TestResult. Also includes
        'optimal_lag' key pointing to the lag with the lowest AIC.
        """
        from statsmodels.tsa.stattools import grangercausalitytests

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        data = np.column_stack([y[mask], x[mask]])
        if len(data) < 2 * max_lag + 10:
            raise ValueError("Insufficient observations for Granger test at max_lag")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        results: dict = {}
        best_aic = np.inf
        best_lag = 1
        for lag, lag_result in raw.items():
            f_stat = float(lag_result[0]["ssr_ftest"][0])
            p_val = float(lag_result[0]["ssr_ftest"][1])
            aic = float(lag_result[1][1].aic)
            if aic < best_aic:
                best_aic = aic
                best_lag = lag
            interp = (
                f"x Granger-causes y at lag {lag} (p={p_val:.4f})" if p_val < 0.05
                else f"x does not Granger-cause y at lag {lag} (p={p_val:.4f})"
            )
            results[lag] = TestResult(
                statistic=f_stat,
                p_value=p_val,
                interpretation=interp,
                extra={"aic": aic},
            )
        results["optimal_lag"] = best_lag
        return results

    @staticmethod
    def white_reality_check(
        strategies: list, benchmark: np.ndarray
    ) -> TestResult:
        """
        White's Reality Check (2000) for data snooping bias.

        Tests whether the best strategy out of a universe outperforms the
        benchmark after correcting for the number of strategies tested.

        Null hypothesis: the best strategy has no predictive superiority over
        the benchmark (all strategies <= benchmark in expected performance).

        Parameters
        ----------
        strategies : list of 1-D return arrays, one per candidate strategy
        benchmark  : benchmark return array (e.g., buy-and-hold)

        Note: we approximate the p-value via stationary bootstrap (l=sqrt(T)).
        """
        strategies = [np.asarray(s, dtype=float) for s in strategies]
        bench = np.asarray(benchmark, dtype=float)
        T = len(bench)

        # performance differentials: f_k(t) = strategy_k(t) - benchmark(t)
        diffs = np.column_stack([s[:T] - bench[:T] for s in strategies])
        f_bar = diffs.mean(axis=0)  # (K,)
        V_max = float(np.sqrt(T) * f_bar.max())  # test statistic

        # Stationary bootstrap to compute p-value
        n_boot = 1000
        rng = np.random.default_rng(seed=0)
        block_len = max(1, int(np.sqrt(T)))
        boot_max = np.zeros(n_boot)

        for b in range(n_boot):
            # draw block-bootstrap indices
            indices = _stationary_bootstrap_indices(T, block_len, rng)
            boot_diffs = diffs[indices, :]
            boot_f_bar = boot_diffs.mean(axis=0)
            # re-center
            centered = np.sqrt(T) * (boot_f_bar - f_bar)
            boot_max[b] = centered.max()

        p_val = float(np.mean(boot_max >= V_max))
        interp = (
            "Best strategy significantly outperforms benchmark (survives reality check)"
            if p_val < 0.05
            else "No strategy significantly outperforms benchmark (data snooping likely)"
        )
        return TestResult(
            statistic=V_max,
            p_value=p_val,
            interpretation=interp,
            extra={"n_strategies": len(strategies), "T": T},
        )

    @staticmethod
    def hansen_spa_test(
        strategies: list, benchmark: np.ndarray
    ) -> TestResult:
        """
        Hansen's Superior Predictive Ability (SPA) test (2005).

        More powerful than White's RC because it uses a studentized test
        statistic and eliminates poorly performing models from the bootstrap
        distribution (consistent SPA).

        Null: no strategy is superior to the benchmark on average.
        """
        strategies = [np.asarray(s, dtype=float) for s in strategies]
        bench = np.asarray(benchmark, dtype=float)
        T = len(bench)

        diffs = np.column_stack([s[:T] - bench[:T] for s in strategies])
        f_bar = diffs.mean(axis=0)
        # studentize
        f_var = diffs.var(axis=0, ddof=1) + 1e-12
        t_stat = float((np.sqrt(T) * f_bar / np.sqrt(f_var)).max())

        n_boot = 1000
        rng = np.random.default_rng(seed=1)
        block_len = max(1, int(np.sqrt(T)))
        boot_t_max = np.zeros(n_boot)

        # consistent SPA: zero out strategies with negative mean (losers)
        mu_c = np.maximum(f_bar, 0.0)

        for b in range(n_boot):
            indices = _stationary_bootstrap_indices(T, block_len, rng)
            boot_diffs = diffs[indices, :]
            boot_f_bar = boot_diffs.mean(axis=0)
            centered = np.sqrt(T) * (boot_f_bar - mu_c) / np.sqrt(f_var)
            boot_t_max[b] = centered.max()

        p_val = float(np.mean(boot_t_max >= t_stat))
        interp = (
            "At least one strategy is significantly superior (SPA test)"
            if p_val < 0.05
            else "No strategy is significantly superior to benchmark (SPA test)"
        )
        return TestResult(
            statistic=t_stat,
            p_value=p_val,
            interpretation=interp,
            extra={"n_strategies": len(strategies), "T": T},
        )


# ---------------------------------------------------------------------------
# Model Diagnostics
# ---------------------------------------------------------------------------

class ModelDiagnostics:
    """
    Residual diagnostic tests for validating statistical model assumptions.
    """

    @staticmethod
    def ljung_box(residuals: np.ndarray, lags: int = 10) -> TestResult:
        """
        Ljung-Box Q test for autocorrelation in residuals.

        Null: residuals are independently distributed (no autocorrelation up
        to the specified lag). Returns the result for the maximum lag tested.
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox

        x = np.asarray(residuals, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < lags + 5:
            raise ValueError(f"Need at least {lags + 5} observations for Ljung-Box")

        result = acorr_ljungbox(x, lags=[lags], return_df=True)
        stat = float(result["lb_stat"].iloc[-1])
        p = float(result["lb_pvalue"].iloc[-1])
        interp = (
            f"Significant autocorrelation at lag {lags} (model residuals not iid)"
            if p < 0.05
            else f"No significant autocorrelation at lag {lags}"
        )
        return TestResult(
            statistic=stat,
            p_value=p,
            interpretation=interp,
            extra={"lags": lags},
        )

    @staticmethod
    def arch_lm_test(residuals: np.ndarray, lags: int = 10) -> TestResult:
        """
        Engle's ARCH LM test for conditional heteroscedasticity.

        Null: no ARCH effects (homoscedastic residuals). A significant result
        indicates GARCH-type volatility clustering.
        """
        from statsmodels.stats.diagnostic import het_arch

        x = np.asarray(residuals, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < lags + 5:
            raise ValueError(f"Need at least {lags + 5} observations for ARCH LM test")

        lm_stat, lm_p, f_stat, f_p = het_arch(x, nlags=lags)
        interp = (
            f"ARCH effects present at lag {lags} (use GARCH model)"
            if lm_p < 0.05
            else f"No ARCH effects detected at lag {lags}"
        )
        return TestResult(
            statistic=float(lm_stat),
            p_value=float(lm_p),
            interpretation=interp,
            extra={"f_stat": float(f_stat), "f_p": float(f_p), "lags": lags},
        )

    @staticmethod
    def breusch_pagan(residuals: np.ndarray, X: np.ndarray) -> TestResult:
        """
        Breusch-Pagan test for heteroscedasticity.

        Regresses squared residuals on X and tests whether the regression
        is significant. Null: homoscedasticity.

        Parameters
        ----------
        residuals : model residuals, shape (n,)
        X         : design matrix, shape (n, k) -- include intercept if desired
        """
        from statsmodels.stats.diagnostic import het_breuschpagan

        e = np.asarray(residuals, dtype=float)
        Xm = np.asarray(X, dtype=float)
        if Xm.ndim == 1:
            Xm = Xm.reshape(-1, 1)

        mask = np.isfinite(e) & np.all(np.isfinite(Xm), axis=1)
        e, Xm = e[mask], Xm[mask]

        lm_stat, lm_p, f_stat, f_p = het_breuschpagan(e, Xm)
        interp = (
            "Heteroscedasticity detected (Breusch-Pagan)" if lm_p < 0.05
            else "Cannot reject homoscedasticity (Breusch-Pagan)"
        )
        return TestResult(
            statistic=float(lm_stat),
            p_value=float(lm_p),
            interpretation=interp,
            extra={"f_stat": float(f_stat), "f_p": float(f_p)},
        )

    @staticmethod
    def durbin_watson(residuals: np.ndarray) -> float:
        """
        Durbin-Watson statistic for first-order serial correlation.

        Returns a scalar in [0, 4].
          ~2: no autocorrelation
          <2: positive autocorrelation
          >2: negative autocorrelation
        Conventional thresholds: <1.5 or >2.5 indicate potential problems.
        """
        from statsmodels.stats.stattools import durbin_watson as sm_dw

        x = np.asarray(residuals, dtype=float)
        x = x[np.isfinite(x)]
        return float(sm_dw(x))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stationary_bootstrap_indices(
    T: int, expected_block_len: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Generate Politis-Romano stationary bootstrap index sequence of length T.

    Each block starts at a uniformly random position and has geometric
    length with mean == expected_block_len.
    """
    p = 1.0 / max(1, expected_block_len)
    indices = np.empty(T, dtype=int)
    i = 0
    while i < T:
        start = rng.integers(0, T)
        block_len = int(rng.geometric(p))
        for j in range(block_len):
            if i >= T:
                break
            indices[i] = (start + j) % T
            i += 1
    return indices
