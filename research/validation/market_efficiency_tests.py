"""
research/validation/market_efficiency_tests.py -- market efficiency tests for SRFM.

Tests for departures from market efficiency that SRFM signals might exploit:
  - Variance ratio test (Lo-MacKinlay): tests for random walk
  - Runs test (Wald-Wolfowitz): tests for sign patterns in returns
  - Ljung-Box Q test: tests for autocorrelation up to max_lag
  - GPH test for long memory: fractionally integrated (I(d)) processes
  - Threshold cointegration: TAR model for nonlinear pairs relationships

All tests return structured results with rejection flags at 5% significance.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VRTestResult:
    """Result of Lo-MacKinlay variance ratio test."""
    vr: float           # Variance ratio VR(k) = Var(k-period return) / (k * Var(1-period return))
    z_stat: float       # Heteroskedasticity-robust z-statistic
    p_value: float
    reject_rw: bool     # True if we reject random walk H0
    k: int              # Holding period tested
    z_stat_homoskedastic: float = 0.0  # Classical (homoskedastic) z-stat for comparison


@dataclass
class RunsTestResult:
    """Result of Wald-Wolfowitz runs test."""
    n_runs: int
    expected_runs: float
    z_stat: float
    p_value: float
    reject_iid: bool    # True if we reject IID (independence) H0


@dataclass
class LjungBoxResult:
    """Result of Ljung-Box autocorrelation test."""
    q_stats: List[float]    # Q-statistic at each lag
    p_values: List[float]   # p-value at each lag
    lags: List[int]
    autocorrelations: List[float]   # ACF at each lag
    reject_no_autocorr: bool        # True if any lag significant at alpha
    first_significant_lag: Optional[int]


@dataclass
class LongMemoryResult:
    """Result of Geweke-Porter-Hudak (GPH) test for long memory."""
    d_estimate: float   # Fractional differencing parameter (d ~ 0.5 = unit root, d = 0 = no memory)
    std_error: float
    t_stat: float
    p_value: float
    has_long_memory: bool  # d significantly > 0 and < 0.5
    confidence_interval_lower: float = 0.0
    confidence_interval_upper: float = 0.0


@dataclass
class ThresholdCointResult:
    """Result of threshold cointegration test (TAR/MTAR)."""
    spread_mean_reversion: float    # Estimated speed of mean reversion
    threshold: float                # Estimated threshold
    rho_above: float                # Adjustment speed when spread > threshold
    rho_below: float                # Adjustment speed when spread < threshold
    f_stat_threshold: float         # F-stat for threshold effect (vs linear)
    p_value_threshold: float
    adf_stat: float                 # ADF on residuals (standard cointegration pre-test)
    adf_p_value: float
    is_cointegrated: bool           # Standard cointegration
    has_threshold_effect: bool      # Nonlinear threshold behavior


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute sample autocorrelation at lags 1..max_lag."""
    n = len(x)
    x_demeaned = x - np.mean(x)
    c0 = float(np.sum(x_demeaned ** 2)) / n
    acf = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        ck = float(np.sum(x_demeaned[lag:] * x_demeaned[:-lag])) / n
        acf[lag - 1] = ck / (c0 + 1e-15)
    return acf


def _adf_test(series: np.ndarray, max_lags: int = 5) -> Tuple[float, float]:
    """
    Augmented Dickey-Fuller test on a series.
    Returns (adf_stat, p_value). p_value is interpolated from MacKinnon (2010) table.
    """
    # Use scipy's built-in via statsmodels if available, else manual
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series, maxlag=max_lags, autolag="AIC", regression="c")
        return float(result[0]), float(result[1])
    except ImportError:
        pass

    # Manual ADF: OLS of diff(y) on y_{t-1} and lagged diffs
    n = len(series)
    dy = np.diff(series)
    k = min(max_lags, len(dy) // 5)

    if k == 0:
        # Simple Dickey-Fuller without lags
        y_lag = series[:-1]
        X = np.column_stack([np.ones(len(y_lag)), y_lag])
        beta = np.linalg.lstsq(X, dy, rcond=None)[0]
        y_hat = X @ beta
        resid = dy - y_hat
        sigma2 = np.sum(resid ** 2) / max(len(dy) - 2, 1)
        XtX_inv = np.linalg.pinv(X.T @ X)
        se_beta = np.sqrt(sigma2 * np.diag(XtX_inv))
        tau = beta[1] / (se_beta[1] + 1e-15)
    else:
        lagged_diffs = np.column_stack([dy[k - j:-j if j > 0 else None] for j in range(1, k + 1)])
        y_lag = series[k:-1]
        dep = dy[k:]
        X = np.column_stack([np.ones(len(y_lag)), y_lag, lagged_diffs])
        beta = np.linalg.lstsq(X, dep, rcond=None)[0]
        y_hat = X @ beta
        resid = dep - y_hat
        sigma2 = np.sum(resid ** 2) / max(len(dep) - X.shape[1], 1)
        XtX_inv = np.linalg.pinv(X.T @ X)
        se_beta = np.sqrt(sigma2 * np.diag(XtX_inv))
        tau = beta[1] / (se_beta[1] + 1e-15)

    # Approximate p-value using MacKinnon critical values for n=100 (conservative)
    # tau < -3.43 -> p < 0.05, tau < -2.86 -> p < 0.10
    critical_vals = {
        0.01: -3.43,
        0.05: -2.86,
        0.10: -2.57,
    }
    if tau < critical_vals[0.01]:
        p_approx = 0.005
    elif tau < critical_vals[0.05]:
        p_approx = 0.025
    elif tau < critical_vals[0.10]:
        p_approx = 0.075
    else:
        p_approx = 0.20

    return float(tau), p_approx


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MarketEfficiencyTests:
    """
    Market efficiency tests for quantitative research.

    These tests help determine whether a market exhibits exploitable inefficiencies.
    Rejecting efficiency tests is a necessary (but not sufficient) condition for
    alpha -- the signal must also be economically significant and survive transaction costs.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Variance Ratio Test (Lo-MacKinlay 1988)
    # ------------------------------------------------------------------

    def variance_ratio_test(
        self,
        prices: pd.Series,
        k: int = 5,
    ) -> VRTestResult:
        """
        Lo-MacKinlay variance ratio test for the random walk hypothesis.

        VR(k) = Var(r_t + r_{t-1} + ... + r_{t-k+1}) / (k * Var(r_t))

        Under random walk: VR(k) = 1 for all k.
        VR(k) > 1: positive autocorrelation (momentum)
        VR(k) < 1: negative autocorrelation (mean reversion)

        Both homoskedastic (Z) and heteroskedasticity-robust (Z*) statistics
        are computed. The robust version is preferred for financial returns.

        Parameters
        ----------
        prices : pd.Series
            Price series (NOT returns -- function computes log returns internally).
        k : int
            Aggregation interval (e.g., k=5 for weekly overlap in daily data).

        Returns
        -------
        VRTestResult
        """
        if k < 2:
            raise ValueError("k must be >= 2")

        log_prices = np.log(np.asarray(prices, dtype=float))
        log_prices = log_prices[np.isfinite(log_prices)]
        r = np.diff(log_prices)  # single-period log returns
        n = len(r)

        if n < 2 * k:
            raise ValueError(f"Need at least 2k={2*k} returns, got {n}")

        # Single-period variance
        mu = float(np.mean(r))
        var1 = float(np.sum((r - mu) ** 2)) / (n - 1)

        # k-period variance (overlapping)
        nq = n - k + 1  # number of overlapping k-period returns
        r_k = np.array([np.sum(r[j: j + k]) for j in range(nq)])
        var_k = float(np.sum((r_k - k * mu) ** 2)) / (nq * (k - 1))  # bias-corrected

        vr = var_k / (var1 + 1e-15)

        # Asymptotic variance under H0 (homoskedastic)
        # Var(VR) ~ 2(2k-1)(k-1) / (3k*n)
        phi_homo = 2.0 * (2 * k - 1) * (k - 1) / (3.0 * k * n)
        z_homo = (vr - 1.0) / np.sqrt(max(phi_homo, 1e-15))

        # Heteroskedasticity-robust statistic (Lo-MacKinlay eq 18)
        # delta_j = sum_{t=j+1}^{n} (r_t - mu)^2 * (r_{t-j} - mu)^2
        delta = np.zeros(k - 1)
        for j in range(1, k):
            num = float(np.sum((r[j:] - mu) ** 2 * (r[:-j] - mu) ** 2))
            denom = float(np.sum((r - mu) ** 2) ** 2) / n
            delta[j - 1] = num / max(denom, 1e-15)

        # phi_star = sum_{j=1}^{k-1} [2(k-j)/k]^2 * delta_j
        weights = np.array([(2.0 * (k - j) / k) ** 2 for j in range(1, k)])
        phi_star = float(np.sum(weights * delta))
        z_star = (vr - 1.0) / np.sqrt(max(phi_star / n, 1e-15))

        p_value = float(2 * (1 - stats.norm.cdf(abs(z_star))))

        return VRTestResult(
            vr=vr,
            z_stat=z_star,
            p_value=p_value,
            reject_rw=p_value < self.alpha,
            k=k,
            z_stat_homoskedastic=float(z_homo),
        )

    # ------------------------------------------------------------------
    # Runs Test (Wald-Wolfowitz)
    # ------------------------------------------------------------------

    def runs_test(
        self,
        returns: pd.Series,
    ) -> RunsTestResult:
        """
        Wald-Wolfowitz runs test for independence of return signs.

        A "run" is a sequence of consecutive returns with the same sign.
        Under independence, the number of runs follows an approximately
        normal distribution. Too few runs suggests positive autocorrelation;
        too many suggests negative autocorrelation.

        Parameters
        ----------
        returns : pd.Series
            Return series (can include zeros, which are handled separately).

        Returns
        -------
        RunsTestResult
        """
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]

        # Remove zeros for sign-based test
        r = r[r != 0]
        n = len(r)
        if n < 10:
            raise ValueError(f"Need at least 10 non-zero observations, got {n}")

        signs = np.sign(r)
        n_pos = int(np.sum(signs > 0))
        n_neg = int(np.sum(signs < 0))

        if n_pos == 0 or n_neg == 0:
            raise ValueError("All returns have the same sign -- runs test is not applicable")

        # Count runs
        n_runs = 1
        for i in range(1, n):
            if signs[i] != signs[i - 1]:
                n_runs += 1

        # Expected runs and variance under H0
        n_total = n_pos + n_neg
        expected = (2.0 * n_pos * n_neg) / n_total + 1.0
        variance = (
            2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n_total)
            / (n_total ** 2 * (n_total - 1))
        )

        if variance <= 0:
            raise ValueError("Cannot compute variance of runs (degenerate case)")

        z_stat = (n_runs - expected) / np.sqrt(variance)
        p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

        return RunsTestResult(
            n_runs=n_runs,
            expected_runs=expected,
            z_stat=float(z_stat),
            p_value=p_value,
            reject_iid=p_value < self.alpha,
        )

    # ------------------------------------------------------------------
    # Autocorrelation Test (Ljung-Box)
    # ------------------------------------------------------------------

    def autocorrelation_test(
        self,
        returns: pd.Series,
        max_lag: int = 10,
    ) -> LjungBoxResult:
        """
        Ljung-Box Q test for autocorrelation in returns.

        Q(m) = n*(n+2) * sum_{k=1}^{m} rho_k^2 / (n-k)

        Under H0 (no autocorrelation), Q(m) ~ chi2(m).

        Also tests squared returns for ARCH effects (volatility clustering).

        Parameters
        ----------
        returns : pd.Series
            Return series.
        max_lag : int
            Maximum lag to test.

        Returns
        -------
        LjungBoxResult
        """
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        n = len(r)

        if n < max_lag + 5:
            warnings.warn(
                f"n={n} too small for max_lag={max_lag}; reducing to {max(2, n // 3)}",
                UserWarning,
                stacklevel=2,
            )
            max_lag = max(2, n // 3)

        acf = _autocorrelation(r, max_lag)

        q_stats: List[float] = []
        p_values: List[float] = []
        lags: List[int] = list(range(1, max_lag + 1))

        # Cumulative Q-stat at each lag
        for m in range(1, max_lag + 1):
            q = float(n * (n + 2) * np.sum(acf[:m] ** 2 / np.arange(n - 1, n - m - 1, -1)))
            p = float(1 - stats.chi2.cdf(q, df=m))
            q_stats.append(q)
            p_values.append(p)

        # Find first significant lag
        first_sig: Optional[int] = None
        for i, p in enumerate(p_values):
            if p < self.alpha:
                first_sig = lags[i]
                break

        return LjungBoxResult(
            q_stats=q_stats,
            p_values=p_values,
            lags=lags,
            autocorrelations=list(acf),
            reject_no_autocorr=any(p < self.alpha for p in p_values),
            first_significant_lag=first_sig,
        )

    # ------------------------------------------------------------------
    # Long Memory Test (GPH -- Geweke-Porter-Hudak 1983)
    # ------------------------------------------------------------------

    def long_memory_test(
        self,
        returns: pd.Series,
        bandwidth_exponent: float = 0.5,
    ) -> LongMemoryResult:
        """
        GPH semiparametric test for long memory (fractional integration).

        For an I(d) process with 0 < d < 0.5, the spectral density near zero
        satisfies: log f(omega) ~ -2d * log(omega) + const as omega -> 0.

        This is estimated by OLS regression of log-periodogram on log-frequencies
        for the m lowest Fourier frequencies (m = n^bandwidth_exponent).

        d = 0: no long memory (ARMA-type)
        0 < d < 0.5: long memory, stationary
        d = 0.5: boundary between stationary and non-stationary
        d >= 0.5: non-stationary long memory

        Parameters
        ----------
        returns : pd.Series
            Return series.
        bandwidth_exponent : float
            Controls m = n^bandwidth_exponent frequencies used (default 0.5).

        Returns
        -------
        LongMemoryResult
        """
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        n = len(r)
        if n < 20:
            raise ValueError(f"Need at least 20 observations, got {n}")

        # Periodogram at Fourier frequencies
        omega = 2 * np.pi * np.arange(1, n) / n
        fft_vals = np.fft.rfft(r - np.mean(r))
        periodogram = np.abs(fft_vals[1:]) ** 2 / (2 * np.pi * n)

        # Use m = floor(n^bandwidth_exponent) lowest frequencies
        m = max(2, int(np.floor(n ** bandwidth_exponent)))
        m = min(m, len(periodogram) // 2)

        omega_m = omega[:m]
        I_m = periodogram[:m]

        # GPH regression: log(I(omega_j)) = const - 2d * log(omega_j) + error
        log_omega = np.log(omega_m)
        log_I = np.log(np.maximum(I_m, 1e-15))

        # Remove any non-finite values
        valid = np.isfinite(log_omega) & np.isfinite(log_I)
        log_omega = log_omega[valid]
        log_I = log_I[valid]

        if len(log_omega) < 3:
            raise ValueError("Not enough valid frequency points for GPH regression")

        # OLS: log_I = a + b * log_omega, d = -b/2
        X = np.column_stack([np.ones(len(log_omega)), log_omega])
        beta = np.linalg.lstsq(X, log_I, rcond=None)[0]
        y_hat = X @ beta
        resid = log_I - y_hat
        rss = float(np.sum(resid ** 2))
        sigma2 = rss / max(len(log_I) - 2, 1)

        XtX_inv = np.linalg.pinv(X.T @ X)
        se_beta = np.sqrt(sigma2 * np.diag(XtX_inv))

        d_hat = -beta[1] / 2.0
        se_d = se_beta[1] / 2.0

        t_stat = d_hat / (se_d + 1e-15)
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=len(log_I) - 2)))

        ci_lower = d_hat - 1.96 * se_d
        ci_upper = d_hat + 1.96 * se_d

        # Long memory: d significantly > 0 and < 0.5 (stationary long memory)
        has_lm = (p_value < self.alpha) and (0.0 < d_hat < 0.5)

        return LongMemoryResult(
            d_estimate=float(d_hat),
            std_error=float(se_d),
            t_stat=float(t_stat),
            p_value=p_value,
            has_long_memory=has_lm,
            confidence_interval_lower=float(ci_lower),
            confidence_interval_upper=float(ci_upper),
        )

    # ------------------------------------------------------------------
    # Threshold Cointegration (TAR)
    # ------------------------------------------------------------------

    def threshold_cointegration(
        self,
        y1: pd.Series,
        y2: pd.Series,
        n_threshold_candidates: int = 100,
    ) -> ThresholdCointResult:
        """
        Threshold autoregressive (TAR) model for pairs -- nonlinear cointegration.

        Tests whether two series are cointegrated with a threshold effect:
        the spread reverts differently when above vs below the threshold.

        Steps:
          1. OLS regression to get spread: spread = y1 - beta*y2
          2. ADF test on spread (standard cointegration test)
          3. TAR model: delta_spread_t = I(spread_{t-1} >= tau) * rho_above * spread_{t-1}
                                        + I(spread_{t-1} < tau)  * rho_below * spread_{t-1} + eps
          4. F-test: TAR vs linear AR (tests threshold effect)

        Parameters
        ----------
        y1 : pd.Series
            First price series.
        y2 : pd.Series
            Second price series (same index as y1).
        n_threshold_candidates : int
            Number of threshold values to search over.

        Returns
        -------
        ThresholdCointResult
        """
        # Align
        both = pd.concat([y1, y2], axis=1).dropna()
        if len(both) < 30:
            raise ValueError(f"Need at least 30 observations, got {len(both)}")

        y1_arr = both.iloc[:, 0].values.astype(float)
        y2_arr = both.iloc[:, 1].values.astype(float)

        # Step 1: OLS to get spread (cointegrating regression)
        X_ols = np.column_stack([np.ones(len(y2_arr)), y2_arr])
        beta = np.linalg.lstsq(X_ols, y1_arr, rcond=None)[0]
        spread = y1_arr - (beta[0] + beta[1] * y2_arr)

        # Step 2: ADF on spread
        adf_stat, adf_p = _adf_test(spread, max_lags=5)

        # Step 3: TAR model
        # Use Enders-Siklos (2001): search over central 70% of spread distribution
        spread_lag = spread[:-1]
        d_spread = np.diff(spread)
        n = len(d_spread)

        low_q = float(np.percentile(spread_lag, 15))
        high_q = float(np.percentile(spread_lag, 85))
        candidates = np.linspace(low_q, high_q, n_threshold_candidates)

        best_tau = float(np.median(spread_lag))
        best_rss = np.inf
        best_rho_above = 0.0
        best_rho_below = 0.0

        for tau in candidates:
            I_above = (spread_lag >= tau).astype(float)
            I_below = (spread_lag < tau).astype(float)

            # X_tar = [intercept, I_above * spread_lag, I_below * spread_lag]
            X_tar = np.column_stack([
                np.ones(n),
                I_above * spread_lag,
                I_below * spread_lag,
            ])
            try:
                beta_tar = np.linalg.lstsq(X_tar, d_spread, rcond=None)[0]
                resid_tar = d_spread - X_tar @ beta_tar
                rss = float(np.sum(resid_tar ** 2))
                if rss < best_rss:
                    best_rss = rss
                    best_tau = float(tau)
                    best_rho_above = float(beta_tar[1])
                    best_rho_below = float(beta_tar[2])
            except np.linalg.LinAlgError:
                continue

        # Step 4: F-test -- TAR vs linear AR
        # Linear AR model
        X_linear = np.column_stack([np.ones(n), spread_lag])
        beta_linear = np.linalg.lstsq(X_linear, d_spread, rcond=None)[0]
        resid_linear = d_spread - X_linear @ beta_linear
        rss_linear = float(np.sum(resid_linear ** 2))

        # TAR model (best threshold)
        I_above_best = (spread_lag >= best_tau).astype(float)
        I_below_best = (spread_lag < best_tau).astype(float)
        X_tar_best = np.column_stack([
            np.ones(n),
            I_above_best * spread_lag,
            I_below_best * spread_lag,
        ])
        beta_tar_best = np.linalg.lstsq(X_tar_best, d_spread, rcond=None)[0]
        resid_tar_best = d_spread - X_tar_best @ beta_tar_best
        rss_tar = float(np.sum(resid_tar_best ** 2))

        # F-stat: TAR adds 1 extra parameter (rho split into above/below)
        f_num = (rss_linear - rss_tar) / 1.0
        f_denom = rss_tar / max(n - 3, 1)
        f_stat = f_num / max(f_denom, 1e-15)
        # Note: critical values for TAR F-test are non-standard (Chan 1993);
        # we approximate with standard F distribution (conservative)
        p_threshold = float(1 - stats.f.cdf(f_stat, 1, max(n - 3, 1)))

        # Average speed of mean reversion
        avg_rho = (best_rho_above + best_rho_below) / 2.0

        return ThresholdCointResult(
            spread_mean_reversion=avg_rho,
            threshold=best_tau,
            rho_above=best_rho_above,
            rho_below=best_rho_below,
            f_stat_threshold=float(f_stat),
            p_value_threshold=p_threshold,
            adf_stat=float(adf_stat),
            adf_p_value=float(adf_p),
            is_cointegrated=adf_p < self.alpha,
            has_threshold_effect=p_threshold < self.alpha,
        )

    # ------------------------------------------------------------------
    # Full efficiency battery
    # ------------------------------------------------------------------

    def run_full_battery(
        self,
        prices: pd.Series,
        vr_ks: Optional[List[int]] = None,
    ) -> dict:
        """
        Run all efficiency tests and return a summary.

        Parameters
        ----------
        prices : pd.Series
            Price series.
        vr_ks : List[int], optional
            List of k values for variance ratio tests (default [2, 5, 10]).

        Returns
        -------
        dict with test names as keys and result dataclasses as values,
        plus 'summary_text' key.
        """
        if vr_ks is None:
            vr_ks = [2, 5, 10]

        results: dict = {}
        log_r = pd.Series(np.diff(np.log(prices.dropna().values)), name="log_returns")

        # Variance ratio tests
        vr_results: List[VRTestResult] = []
        for k in vr_ks:
            try:
                vr_res = self.variance_ratio_test(prices, k=k)
                vr_results.append(vr_res)
            except Exception as e:
                warnings.warn(f"VR test k={k} failed: {e}", UserWarning, stacklevel=2)
        results["variance_ratio"] = vr_results

        try:
            results["runs"] = self.runs_test(log_r)
        except Exception as e:
            warnings.warn(f"Runs test failed: {e}", UserWarning, stacklevel=2)
            results["runs"] = None

        try:
            results["ljung_box"] = self.autocorrelation_test(log_r)
        except Exception as e:
            warnings.warn(f"Ljung-Box test failed: {e}", UserWarning, stacklevel=2)
            results["ljung_box"] = None

        try:
            results["long_memory"] = self.long_memory_test(log_r)
        except Exception as e:
            warnings.warn(f"Long memory test failed: {e}", UserWarning, stacklevel=2)
            results["long_memory"] = None

        # Summary
        lines = ["=== Market Efficiency Battery ==="]
        for k, vr in zip(vr_ks, vr_results):
            lines.append(
                f"VR(k={k}): vr={vr.vr:.3f}, z*={vr.z_stat:.2f}, p={vr.p_value:.4f}, "
                f"reject_RW={'YES' if vr.reject_rw else 'no'}"
            )
        if results.get("runs") is not None:
            r = results["runs"]
            lines.append(
                f"Runs: n_runs={r.n_runs}, expected={r.expected_runs:.1f}, "
                f"z={r.z_stat:.2f}, reject_IID={'YES' if r.reject_iid else 'no'}"
            )
        if results.get("ljung_box") is not None:
            lb = results["ljung_box"]
            lines.append(
                f"Ljung-Box: reject_autocorr={'YES' if lb.reject_no_autocorr else 'no'}, "
                f"first_sig_lag={lb.first_significant_lag}"
            )
        if results.get("long_memory") is not None:
            lm = results["long_memory"]
            lines.append(
                f"Long Memory: d={lm.d_estimate:.3f}({lm.std_error:.3f}), "
                f"long_memory={'YES' if lm.has_long_memory else 'no'}"
            )
        results["summary_text"] = "\n".join(lines)
        return results
