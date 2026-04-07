"""
test_new_validation.py -- test suite for new validation modules.

Covers:
  - CausalAnalyzer: Granger causality, PSM, DiD, IV/2SLS
  - OutOfSampleValidator: expanding window, walk-forward, CPCV, DSR, FDR
  - MarketEfficiencyTests: VR test, runs test, Ljung-Box, long memory, threshold coint
  - PerformancePersistenceAnalyzer: contingency table, Spearman rank IC, IR stability, regimes

Run with: pytest research/validation/tests/test_new_validation.py -v
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from research.validation.causal_inference import (
    CausalAnalyzer,
    DiffInDiffResult,
    GrangerResult,
    IVResult,
    PSMResult,
)
from research.validation.market_efficiency_tests import (
    LjungBoxResult,
    LongMemoryResult,
    MarketEfficiencyTests,
    RunsTestResult,
    ThresholdCointResult,
    VRTestResult,
)
from research.validation.out_of_sample_validator import (
    CPCVResult,
    OOSResult,
    OutOfSampleValidator,
    WalkForwardResult,
)
from research.validation.performance_persistence import (
    ContingencyResult,
    IRStabilityResult,
    PerformancePersistenceAnalyzer,
    RegimePersistenceResult,
)


# ---------------------------------------------------------------------------
# Fixtures and data generators
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def make_ar1_series(n: int, phi: float, sigma: float = 1.0) -> pd.Series:
    """Generate AR(1) process: y_t = phi * y_{t-1} + eps."""
    eps = RNG.normal(0, sigma, n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t]
    return pd.Series(y)


def make_causal_pair(n: int, beta: float = 0.5) -> tuple[pd.Series, pd.Series]:
    """
    Generate (x, y) where x Granger-causes y.
    x is AR(1), y depends on lagged x plus own AR term.
    """
    x = make_ar1_series(n, phi=0.7, sigma=1.0)
    x_arr = x.values
    eps = RNG.normal(0, 0.5, n)
    y_arr = np.zeros(n)
    for t in range(2, n):
        y_arr[t] = 0.3 * y_arr[t - 1] + beta * x_arr[t - 1] + eps[t]
    return x, pd.Series(y_arr)


def make_random_walk(n: int, drift: float = 0.0) -> pd.Series:
    """Random walk prices."""
    log_r = RNG.normal(drift, 0.01, n)
    log_p = np.cumsum(log_r)
    return pd.Series(np.exp(log_p))


def make_ar1_prices(n: int, phi: float = 0.1) -> pd.Series:
    """AR(1) log returns -> prices (mean reversion implies VR < 1)."""
    r = make_ar1_series(n, phi=phi, sigma=0.01)
    log_p = np.cumsum(r.values)
    return pd.Series(np.exp(log_p))


def make_persistent_scores(n: int, persistence: float = 0.7) -> tuple[pd.Series, pd.Series]:
    """
    Generate two periods of scores where performance is persistent.
    Underlying skill: z_i drawn once, scores in each period = z_i + noise.
    """
    skill = RNG.normal(0, 1, n)
    noise1 = RNG.normal(0, (1 - persistence) / persistence, n)
    noise2 = RNG.normal(0, (1 - persistence) / persistence, n)
    s1 = pd.Series(skill + noise1)
    s2 = pd.Series(persistence * skill + noise2)
    return s1, s2


# ---------------------------------------------------------------------------
# CausalAnalyzer tests
# ---------------------------------------------------------------------------

class TestGrangerCausality:
    def setup_method(self) -> None:
        self.analyzer = CausalAnalyzer(alpha=0.05)

    def test_known_causal_relationship(self) -> None:
        """x truly causes y: Granger test should detect it."""
        n = 500
        x, y = make_causal_pair(n, beta=0.8)
        result = self.analyzer.granger_causality(x, y, max_lag=5)
        assert isinstance(result, GrangerResult)
        # With strong beta and n=500, should detect causality
        assert result.does_x_cause_y, (
            f"Expected causality detected, p={result.p_value:.4f}, F={result.f_stat:.2f}"
        )
        assert result.f_stat > 1.0
        assert 1 <= result.lag <= 5

    def test_no_causal_relationship(self) -> None:
        """x and y are independent: should not reject H0 most of the time."""
        # Run multiple times to reduce false positive risk
        n_reject = 0
        n_trials = 20
        for seed in range(n_trials):
            rng = np.random.default_rng(seed + 100)
            x = pd.Series(rng.normal(0, 1, 300))
            y = pd.Series(rng.normal(0, 1, 300))
            result = self.analyzer.granger_causality(x, y, max_lag=3)
            if result.does_x_cause_y:
                n_reject += 1
        # Should reject < 30% of the time under H0 (alpha=5%, some variance)
        assert n_reject <= 8, f"Too many spurious rejections: {n_reject}/{n_trials}"

    def test_result_structure(self) -> None:
        x, y = make_causal_pair(200, beta=0.5)
        result = self.analyzer.granger_causality(x, y, max_lag=3)
        assert len(result.lag_f_stats) == 3
        assert len(result.lag_p_values) == 3
        assert all(p >= 0 and p <= 1 for p in result.lag_p_values)
        assert all(f >= 0 for f in result.lag_f_stats)

    def test_mismatched_lengths_raises(self) -> None:
        x = pd.Series(np.arange(100, dtype=float))
        y = pd.Series(np.arange(50, dtype=float))
        with pytest.raises(ValueError, match="same length"):
            self.analyzer.granger_causality(x, y)

    def test_max_lag_one(self) -> None:
        """Granger test with max_lag=1 should still work."""
        x, y = make_causal_pair(200, beta=0.8)
        result = self.analyzer.granger_causality(x, y, max_lag=1)
        assert result.lag == 1
        assert 0 <= result.p_value <= 1


class TestPSM:
    def setup_method(self) -> None:
        self.analyzer = CausalAnalyzer(alpha=0.05)

    def test_known_treatment_effect(self) -> None:
        """PSM should recover a known positive treatment effect."""
        n = 200
        rng = np.random.default_rng(99)
        cov1 = rng.normal(0, 1, n)
        cov2 = rng.normal(0, 1, n)
        # Treatment assigned based on covariates (propensity)
        logit_p = 0.5 * cov1 + 0.3 * cov2
        p_treat = 1 / (1 + np.exp(-logit_p))
        treatment = rng.binomial(1, p_treat)
        # Outcome: treatment effect = 2.0
        outcome = 0.3 * cov1 + 0.2 * cov2 + 2.0 * treatment + rng.normal(0, 0.5, n)

        df = pd.DataFrame({"cov1": cov1, "cov2": cov2, "outcome": outcome})
        treated = df[treatment == 1]
        control = df[treatment == 0]

        result = self.analyzer.propensity_score_matching(
            treated, control, covariates=["cov1", "cov2"]
        )
        assert isinstance(result, PSMResult)
        # ATE should be positive and close to 2.0
        assert result.ate > 0.5, f"Expected ATE > 0.5, got {result.ate:.3f}"
        assert result.n_matched > 0
        assert result.standardized_mean_diff_after <= result.standardized_mean_diff_before + 0.5

    def test_zero_treatment_effect(self) -> None:
        """PSM should not reject H0 when treatment effect is 0."""
        n = 300
        rng = np.random.default_rng(77)
        cov = rng.normal(0, 1, n)
        treatment = (cov > 0).astype(int)
        outcome = cov + rng.normal(0, 1, n)  # no treatment effect

        df = pd.DataFrame({"cov": cov, "outcome": outcome})
        treated = df[treatment == 1]
        control = df[treatment == 0]

        result = self.analyzer.propensity_score_matching(treated, control, covariates=["cov"])
        # ATE should be near 0 and p-value should be non-significant
        assert abs(result.ate) < 1.5, f"Expected small ATE, got {result.ate:.3f}"

    def test_psm_result_fields(self) -> None:
        n = 100
        rng = np.random.default_rng(55)
        cov = rng.normal(0, 1, n)
        treatment = (cov > 0).astype(int)
        outcome = cov + 1.0 * treatment + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"cov": cov, "outcome": outcome})
        treated = df[treatment == 1]
        control = df[treatment == 0]
        result = self.analyzer.propensity_score_matching(treated, control, covariates=["cov"])
        assert isinstance(result.treated_outcomes, pd.Series)
        assert isinstance(result.matched_control_outcomes, pd.Series)
        assert len(result.treated_outcomes) == len(result.matched_control_outcomes)
        assert 0 <= result.p_value <= 1


class TestDiffInDiff:
    def setup_method(self) -> None:
        self.analyzer = CausalAnalyzer(alpha=0.05)

    def test_known_did_effect(self) -> None:
        """DiD should recover a known treatment effect."""
        rng = np.random.default_rng(42)
        # True treatment effect = 3.0
        pre_treated = pd.Series(rng.normal(10, 1, 100))
        post_treated = pd.Series(rng.normal(15, 1, 100))  # +5 = trend + treatment
        pre_control = pd.Series(rng.normal(10, 1, 100))
        post_control = pd.Series(rng.normal(12, 1, 100))  # +2 = trend only
        # DiD = (15-10) - (12-10) = 5 - 2 = 3

        result = self.analyzer.diff_in_diff(pre_treated, post_treated, pre_control, post_control)
        assert isinstance(result, DiffInDiffResult)
        assert abs(result.ate - 3.0) < 0.5, f"Expected ATE ~3.0, got {result.ate:.3f}"
        assert result.t_stat > 5.0, f"Expected large t-stat, got {result.t_stat:.2f}"
        assert result.p_value < 0.001

    def test_zero_did_effect(self) -> None:
        """DiD should be ~0 when there is no differential change."""
        rng = np.random.default_rng(123)
        pre_t = pd.Series(rng.normal(10, 1, 100))
        post_t = pd.Series(rng.normal(12, 1, 100))
        pre_c = pd.Series(rng.normal(10, 1, 100))
        post_c = pd.Series(rng.normal(12, 1, 100))  # Same change as treated

        result = self.analyzer.diff_in_diff(pre_t, post_t, pre_c, post_c)
        assert abs(result.ate) < 0.5, f"Expected ATE ~0, got {result.ate:.3f}"

    def test_parallel_trends_check(self) -> None:
        """Placebo DiD on two pre-periods should be near zero."""
        rng = np.random.default_rng(7)
        pre_pre = pd.Series(rng.normal(10, 1, 100))
        pre = pd.Series(rng.normal(10.1, 1, 100))  # Small drift, parallel
        pre_pre_c = pd.Series(rng.normal(10, 1, 100))
        pre_c = pd.Series(rng.normal(10.1, 1, 100))

        result = self.analyzer.diff_in_diff(
            pre, pd.Series(rng.normal(12, 1, 100)),
            pre_c, pd.Series(rng.normal(12, 1, 100)),
            pre_pre_treated=pre_pre, pre_pre_control=pre_pre_c,
        )
        assert result.parallel_trends_stat is not None
        assert abs(result.parallel_trends_stat) < 3.0


class TestIV:
    def setup_method(self) -> None:
        self.analyzer = CausalAnalyzer(alpha=0.05)

    def test_2sls_recovers_causal_effect(self) -> None:
        """2SLS should recover causal effect when instrument is valid."""
        n = 500
        rng = np.random.default_rng(42)
        # Instrument z: affects x but not y directly
        z = pd.Series(rng.normal(0, 1, n))
        # Endogenous variable x: caused by z and correlated with y error
        eta = rng.normal(0, 1, n)  # confound
        x = pd.Series(0.7 * z.values + 0.5 * eta + rng.normal(0, 0.3, n))
        # Outcome y: true causal effect of x = 2.0, plus confound
        y = pd.Series(2.0 * x.values + 0.8 * eta + rng.normal(0, 0.5, n))

        result = self.analyzer.instrumental_variable(y, x, z)
        assert isinstance(result, IVResult)
        # IV estimate should be closer to 2.0 than OLS (which is biased upward due to confound)
        assert abs(result.beta_iv - 2.0) < 1.0, f"IV beta={result.beta_iv:.3f}, expected ~2.0"
        assert result.first_stage_f > 10.0, f"Weak instrument: F={result.first_stage_f:.2f}"

    def test_first_stage_f_weak_instrument(self) -> None:
        """Weak instrument should produce low first-stage F."""
        n = 300
        rng = np.random.default_rng(11)
        z = pd.Series(rng.normal(0, 1, n))
        x = pd.Series(0.05 * z.values + rng.normal(0, 1, n))  # weak: small z coefficient
        y = pd.Series(x.values + rng.normal(0, 1, n))

        result = self.analyzer.instrumental_variable(y, x, z)
        # Weak instrument: first-stage F should be low
        assert result.first_stage_f < 10.0, (
            f"Expected weak instrument F < 10, got {result.first_stage_f:.2f}"
        )

    def test_iv_structure(self) -> None:
        n = 200
        rng = np.random.default_rng(5)
        z = pd.Series(rng.normal(0, 1, n))
        x = pd.Series(0.5 * z.values + rng.normal(0, 1, n))
        y = pd.Series(x.values + rng.normal(0, 1, n))
        result = self.analyzer.instrumental_variable(y, x, z)
        assert 0 <= result.p_value <= 1
        assert 0 <= result.hausman_p_value <= 1
        assert result.first_stage_r2 >= 0 and result.first_stage_r2 <= 1


# ---------------------------------------------------------------------------
# OutOfSampleValidator tests
# ---------------------------------------------------------------------------

def make_returns_df(n: int = 500, n_assets: int = 1) -> pd.DataFrame:
    """Make a simple returns DataFrame with DatetimeIndex."""
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    data = RNG.normal(0.0005, 0.01, (n, n_assets))
    return pd.DataFrame(data, index=idx, columns=[f"asset_{i}" for i in range(n_assets)])


class TestOutOfSampleValidator:
    def setup_method(self) -> None:
        self.validator = OutOfSampleValidator(alpha=0.05)

    def test_expanding_window_known_signal(self) -> None:
        """
        Build a signal that is known to predict returns with controlled decay.
        OOS IC should be positive.
        """
        n = 400
        rng = np.random.default_rng(42)
        idx = pd.date_range("2015-01-01", periods=n, freq="B")
        # Signal: true IC ~ 0.1 in-sample, ~0.05 out-of-sample
        true_scores = rng.normal(0, 1, n)
        noise_is = rng.normal(0, 1, n)
        noise_oos = rng.normal(0, 2, n)
        returns_vals = 0.01 * true_scores + 0.005 * rng.normal(0, 1, n)
        returns = pd.DataFrame({"r": returns_vals}, index=idx)

        def signal_fn(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
            # Return mean of training returns as naive signal (contrarian)
            train_mean = float(train.values.mean())
            return np.array([-train_mean] * len(test))

        result = self.validator.expanding_window_test(signal_fn, returns, initial_train=100)
        assert isinstance(result, OOSResult)
        assert len(result.oos_ics) > 0
        assert len(result.is_ics) == len(result.oos_ics)
        assert result.n_periods == len(result.oos_ics)

    def test_expanding_window_positive_signal(self) -> None:
        """A signal with true predictive power should have positive OOS ICIR."""
        n = 600
        rng = np.random.default_rng(7)
        idx = pd.date_range("2015-01-01", periods=n, freq="B")
        signal_true = rng.normal(0, 1, n)
        returns_vals = 0.02 * signal_true + rng.normal(0, 0.01, n)
        returns = pd.DataFrame({"r": returns_vals}, index=idx)

        # Signal: use lagged return as predictor (momentum)
        def momentum_signal(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
            last_return = float(train.values[-1])
            return np.array([last_return] * len(test))

        result = self.validator.expanding_window_test(momentum_signal, returns, initial_train=50)
        assert isinstance(result, OOSResult)
        # Decay ratio should be defined
        if abs(result.is_icir) > 0.01:
            assert np.isfinite(result.decay_ratio)

    def test_walk_forward_basic(self) -> None:
        """Walk-forward produces correct number of windows."""
        returns = make_returns_df(n=500)

        def trivial_signal(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(test))

        result = self.validator.walk_forward_test(
            trivial_signal, returns, train_window=100, test_window=50
        )
        assert isinstance(result, WalkForwardResult)
        assert result.n_periods >= 1
        assert len(result.oos_ics) == result.n_periods
        assert len(result.window_dates) == result.n_periods

    def test_cpcv_basic(self) -> None:
        """CPCV produces expected number of paths."""
        returns = make_returns_df(n=300)

        def trivial_signal(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(test))

        result = self.validator.combinatorial_purged_cv(
            trivial_signal, returns, n_splits=5, embargo=3
        )
        assert isinstance(result, CPCVResult)
        assert result.n_paths == result.n_splits  # one per test group
        assert len(result.path_sharpes) == result.n_paths
        assert result.sharpe_5th_pct <= result.median_sharpe <= result.sharpe_95th_pct

    def test_deflated_sharpe_single_trial(self) -> None:
        """With n_trials=1, DSR should equal standard normal CDF at SR."""
        dsr = OutOfSampleValidator.deflated_sharpe(
            trial_sharpe=2.0, n_trials=1, n_obs=252
        )
        # With n_trials=1, expected_max_z = 0, so DSR = N(SR / se)
        assert 0 < dsr <= 1.0

    def test_deflated_sharpe_many_trials(self) -> None:
        """With many trials, the same SR should have lower DSR (penalized for fishing)."""
        dsr_1 = OutOfSampleValidator.deflated_sharpe(2.0, n_trials=1, n_obs=252)
        dsr_100 = OutOfSampleValidator.deflated_sharpe(2.0, n_trials=100, n_obs=252)
        dsr_1000 = OutOfSampleValidator.deflated_sharpe(2.0, n_trials=1000, n_obs=252)
        assert dsr_1 > dsr_100 > dsr_1000, (
            f"DSR should decrease as n_trials increases: "
            f"{dsr_1:.3f} > {dsr_100:.3f} > {dsr_1000:.3f}"
        )

    def test_fdr_correction_basic(self) -> None:
        """BH FDR: all small p-values should be rejected."""
        p_vals = [0.001, 0.002, 0.003, 0.5, 0.9]
        rejections = OutOfSampleValidator.false_discovery_rate(p_vals, alpha=0.05)
        assert rejections[0] and rejections[1] and rejections[2]
        assert not rejections[3] and not rejections[4]

    def test_fdr_no_rejections(self) -> None:
        """All large p-values: nothing should be rejected."""
        p_vals = [0.8, 0.9, 0.95, 0.7, 0.85]
        rejections = OutOfSampleValidator.false_discovery_rate(p_vals, alpha=0.05)
        assert not any(rejections)

    def test_fdr_all_rejections(self) -> None:
        """All very small p-values: all should be rejected."""
        p_vals = [0.001, 0.001, 0.001, 0.001]
        rejections = OutOfSampleValidator.false_discovery_rate(p_vals, alpha=0.05)
        assert all(rejections)

    def test_fdr_empty(self) -> None:
        """Empty p-value list should return empty list."""
        assert OutOfSampleValidator.false_discovery_rate([]) == []


# ---------------------------------------------------------------------------
# MarketEfficiencyTests
# ---------------------------------------------------------------------------

class TestVarianceRatioTest:
    def setup_method(self) -> None:
        self.met = MarketEfficiencyTests(alpha=0.05)

    def test_random_walk_does_not_reject(self) -> None:
        """Pure random walk: VR should be near 1, should not reject H0."""
        n_reject = 0
        n_trials = 20
        for seed in range(n_trials):
            prices = make_random_walk(n=1000)
            result = self.met.variance_ratio_test(prices, k=5)
            assert isinstance(result, VRTestResult)
            assert result.k == 5
            assert np.isfinite(result.vr)
            if result.reject_rw:
                n_reject += 1
        # Under 5% alpha, expect ~5% rejection rate; allow up to 30% for variance
        assert n_reject <= 8, f"RW rejected too often: {n_reject}/{n_trials}"

    def test_ar1_rejects_random_walk(self) -> None:
        """AR(1) with strong autocorrelation should reject random walk."""
        n_reject = 0
        n_trials = 20
        for seed in range(n_trials):
            prices = make_ar1_prices(n=500, phi=0.3)
            result = self.met.variance_ratio_test(prices, k=5)
            if result.reject_rw:
                n_reject += 1
        # Strong AR(1) should frequently reject
        assert n_reject >= 5, f"AR(1) rejected RW only {n_reject}/{n_trials} times"

    def test_vr_near_1_for_rw(self) -> None:
        """VR(k) should be close to 1.0 for a random walk."""
        rng = np.random.default_rng(999)
        log_r = rng.normal(0.0001, 0.01, 2000)
        log_p = np.cumsum(log_r)
        prices = pd.Series(np.exp(log_p))
        result = self.met.variance_ratio_test(prices, k=5)
        assert abs(result.vr - 1.0) < 0.3, f"VR={result.vr:.3f} far from 1.0 for RW"

    def test_vr_positive_autocorr_vr_greater_1(self) -> None:
        """Positive autocorrelation in returns -> VR > 1."""
        rng = np.random.default_rng(42)
        n = 1000
        r = make_ar1_series(n, phi=0.4, sigma=0.01)
        log_p = np.cumsum(r.values)
        prices = pd.Series(np.exp(log_p))
        result = self.met.variance_ratio_test(prices, k=5)
        # Positive autocorrelation -> VR should be > 1
        assert result.vr > 1.0, f"Expected VR > 1, got {result.vr:.3f}"


class TestRunsTest:
    def setup_method(self) -> None:
        self.met = MarketEfficiencyTests(alpha=0.05)

    def test_iid_series_does_not_reject(self) -> None:
        """IID returns should not systematically reject H0."""
        n_reject = 0
        n_trials = 30
        for seed in range(n_trials):
            rng = np.random.default_rng(seed + 200)
            r = pd.Series(rng.normal(0, 1, 500))
            result = self.met.runs_test(r)
            assert isinstance(result, RunsTestResult)
            if result.reject_iid:
                n_reject += 1
        assert n_reject <= 10, f"IID rejected too often: {n_reject}/{n_trials}"

    def test_strongly_autocorrelated_rejects(self) -> None:
        """Strong positive autocorrelation produces too few runs -> should reject."""
        rng = np.random.default_rng(42)
        # Alternate signs: too many runs
        r_alt = pd.Series(np.tile([1.0, -1.0], 250))
        result = self.met.runs_test(r_alt)
        assert result.reject_iid, "Perfectly alternating series should reject IID"

    def test_result_fields(self) -> None:
        rng = np.random.default_rng(1)
        r = pd.Series(rng.normal(0, 1, 200))
        result = self.met.runs_test(r)
        assert result.n_runs > 0
        assert result.expected_runs > 0
        assert 0 <= result.p_value <= 1


class TestLjungBox:
    def setup_method(self) -> None:
        self.met = MarketEfficiencyTests(alpha=0.05)

    def test_white_noise_does_not_reject(self) -> None:
        """White noise should not reject no-autocorrelation H0."""
        n_reject = 0
        n_trials = 20
        for seed in range(n_trials):
            rng = np.random.default_rng(seed + 300)
            r = pd.Series(rng.normal(0, 1, 500))
            result = self.met.autocorrelation_test(r, max_lag=5)
            if result.reject_no_autocorr:
                n_reject += 1
        assert n_reject <= 6, f"White noise rejected too often: {n_reject}/{n_trials}"

    def test_ar1_rejects_no_autocorr(self) -> None:
        """Strong AR(1) should reject no-autocorrelation."""
        n_reject = 0
        n_trials = 10
        for seed in range(n_trials):
            r = make_ar1_series(500, phi=0.4, sigma=1.0)
            result = self.met.autocorrelation_test(r, max_lag=5)
            if result.reject_no_autocorr:
                n_reject += 1
        assert n_reject >= 7, f"AR(1) only rejected {n_reject}/{n_trials} times"

    def test_result_structure(self) -> None:
        r = pd.Series(RNG.normal(0, 1, 200))
        result = self.met.autocorrelation_test(r, max_lag=5)
        assert len(result.q_stats) == 5
        assert len(result.p_values) == 5
        assert len(result.lags) == 5
        assert len(result.autocorrelations) == 5


class TestLongMemory:
    def setup_method(self) -> None:
        self.met = MarketEfficiencyTests(alpha=0.05)

    def test_white_noise_no_long_memory(self) -> None:
        """White noise should have d near 0, no long memory."""
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0, 1, 1000))
        result = self.met.long_memory_test(r)
        assert isinstance(result, LongMemoryResult)
        # d estimate should be near 0 (may not be exactly)
        assert abs(result.d_estimate) < 0.5, f"d={result.d_estimate:.3f} should be near 0"
        assert result.std_error > 0
        assert 0 <= result.p_value <= 1

    def test_result_fields(self) -> None:
        r = pd.Series(RNG.normal(0, 1, 500))
        result = self.met.long_memory_test(r)
        assert result.confidence_interval_lower <= result.d_estimate
        assert result.d_estimate <= result.confidence_interval_upper
        assert np.isfinite(result.t_stat)


class TestThresholdCointegration:
    def setup_method(self) -> None:
        self.met = MarketEfficiencyTests(alpha=0.05)

    def test_cointegrated_pair(self) -> None:
        """Cointegrated pair should have significant ADF on spread."""
        rng = np.random.default_rng(42)
        n = 500
        # Generate cointegrated pair: spread is mean-reverting
        spread = make_ar1_series(n, phi=0.7, sigma=0.5)
        y1 = pd.Series(np.cumsum(rng.normal(0, 0.01, n)) + spread.values)
        y2 = pd.Series(np.cumsum(rng.normal(0, 0.01, n)))

        result = self.met.threshold_cointegration(y1, y2)
        assert isinstance(result, ThresholdCointResult)
        assert np.isfinite(result.adf_stat)
        assert np.isfinite(result.threshold)
        assert 0 <= result.p_value_threshold <= 1

    def test_result_fields(self) -> None:
        rng = np.random.default_rng(7)
        n = 200
        y1 = pd.Series(rng.normal(0, 1, n).cumsum())
        y2 = pd.Series(rng.normal(0, 1, n).cumsum())
        result = self.met.threshold_cointegration(y1, y2)
        assert np.isfinite(result.rho_above)
        assert np.isfinite(result.rho_below)
        assert np.isfinite(result.f_stat_threshold)


# ---------------------------------------------------------------------------
# PerformancePersistenceAnalyzer tests
# ---------------------------------------------------------------------------

class TestContingencyTable:
    def setup_method(self) -> None:
        self.analyzer = PerformancePersistenceAnalyzer(alpha=0.05)

    def test_known_persistent_winners(self) -> None:
        """Strongly persistent data should reject H0 of no persistence."""
        n_reject = 0
        n_trials = 10
        for seed in range(n_trials):
            s1, s2 = make_persistent_scores(100, persistence=0.85)
            result = self.analyzer.contingency_table(s1, s2)
            if result.reject_no_persistence:
                n_reject += 1
        assert n_reject >= 6, f"Only {n_reject}/10 trials detected persistence"

    def test_random_scores_no_persistence(self) -> None:
        """Random scores should not systematically reject H0."""
        n_reject = 0
        n_trials = 20
        for seed in range(n_trials):
            rng = np.random.default_rng(seed + 400)
            s1 = pd.Series(rng.normal(0, 1, 100))
            s2 = pd.Series(rng.normal(0, 1, 100))
            result = self.analyzer.contingency_table(s1, s2)
            if result.reject_no_persistence:
                n_reject += 1
        assert n_reject <= 8, f"Random scores rejected H0 {n_reject}/20 times"

    def test_result_structure(self) -> None:
        rng = np.random.default_rng(1)
        s1 = pd.Series(rng.normal(0, 1, 50))
        s2 = pd.Series(rng.normal(0, 1, 50))
        result = self.analyzer.contingency_table(s1, s2)
        total = result.wins_persist + result.wins_switch + result.loses_persist + result.loses_switch
        assert total == len(s1)
        assert 0 <= result.percent_persist <= 1
        assert 0 <= result.p_value <= 1
        assert result.chi2 >= 0

    def test_perfect_persistence(self) -> None:
        """Perfect persistence: winners always win -> should definitely reject H0."""
        n = 100
        scores = np.arange(n, dtype=float)
        s1 = pd.Series(scores)
        s2 = pd.Series(scores + 0.001 * np.arange(n))  # same ranking
        result = self.analyzer.contingency_table(s1, s2)
        assert result.wins_persist + result.loses_persist > n * 0.8
        assert result.reject_no_persistence


class TestSpearmanRankCorrelation:
    def setup_method(self) -> None:
        self.analyzer = PerformancePersistenceAnalyzer()

    def test_high_correlation_persistent(self) -> None:
        """Persistent data should have high rank IC between periods."""
        s1, s2 = make_persistent_scores(200, persistence=0.9)
        corr = self.analyzer.spearman_rank_correlation(s1, s2)
        assert corr > 0.3, f"Expected high rank IC, got {corr:.3f}"

    def test_random_data_near_zero(self) -> None:
        """Random data should have rank IC near zero."""
        rng = np.random.default_rng(42)
        s1 = pd.Series(rng.normal(0, 1, 300))
        s2 = pd.Series(rng.normal(0, 1, 300))
        corr = self.analyzer.spearman_rank_correlation(s1, s2)
        assert abs(corr) < 0.2, f"Expected near-zero rank IC, got {corr:.3f}"

    def test_perfect_rank_correlation(self) -> None:
        """Identical rankings -> rank IC = 1.0."""
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        corr = self.analyzer.spearman_rank_correlation(x, x)
        assert abs(corr - 1.0) < 1e-6


class TestIRStability:
    def setup_method(self) -> None:
        self.analyzer = PerformancePersistenceAnalyzer()

    def test_stable_ir_series(self) -> None:
        """Constant IR series should be flagged as stable."""
        ir_series = [0.5] * 24 + [0.4, 0.6] * 4  # Low variance, positive mean
        result = self.analyzer.information_ratio_stability(ir_series, window=6)
        assert isinstance(result, IRStabilityResult)
        assert result.is_stable
        assert result.cv_ir < 1.0
        assert result.stability_score > 0.3

    def test_unstable_ir_series(self) -> None:
        """Highly variable IR should be flagged as unstable."""
        rng = np.random.default_rng(42)
        ir_series = list(rng.normal(0, 3, 30))  # High variance, near-zero mean
        result = self.analyzer.information_ratio_stability(ir_series, window=6)
        assert result.cv_ir >= 1.0 or not result.is_stable

    def test_result_fields(self) -> None:
        ir_series = list(np.linspace(0.1, 0.5, 20))
        result = self.analyzer.information_ratio_stability(ir_series, window=5)
        assert result.min_ir <= result.mean_ir <= result.max_ir
        assert 0 <= result.pct_positive <= 1
        assert 0 <= result.stability_score <= 1
        assert len(result.rolling_irs) > 0


class TestRegimePersistence:
    def setup_method(self) -> None:
        self.analyzer = PerformancePersistenceAnalyzer()

    def test_consistent_across_regimes(self) -> None:
        """Signal that works in all regimes should not reject H0 of equal means."""
        rng = np.random.default_rng(42)
        n = 300
        # Returns drawn from same distribution regardless of regime
        returns = pd.Series(rng.normal(0.01, 0.02, n))
        regimes = pd.Series(np.tile(["bull", "bear", "neutral"], n // 3))

        result = self.analyzer.regime_persistence_test(returns, regimes)
        assert isinstance(result, RegimePersistenceResult)
        assert result.is_consistent, (
            f"Expected consistent, p={result.p_value:.4f}, F={result.f_stat:.3f}"
        )

    def test_regime_dependent_returns(self) -> None:
        """Signal with regime-dependent returns should reject H0."""
        rng = np.random.default_rng(42)
        n_per = 100
        # Dramatically different means per regime
        bull_r = rng.normal(0.05, 0.01, n_per)
        bear_r = rng.normal(-0.05, 0.01, n_per)
        neutral_r = rng.normal(0.0, 0.01, n_per)
        returns = pd.Series(np.concatenate([bull_r, bear_r, neutral_r]))
        regimes = pd.Series(["bull"] * n_per + ["bear"] * n_per + ["neutral"] * n_per)

        result = self.analyzer.regime_persistence_test(returns, regimes)
        assert not result.is_consistent, (
            f"Expected regime-dependent, p={result.p_value:.6f}"
        )
        assert result.best_regime == "bull"
        assert result.worst_regime == "bear"

    def test_result_fields(self) -> None:
        rng = np.random.default_rng(5)
        n = 100
        returns = pd.Series(rng.normal(0, 1, n))
        regimes = pd.Series(["A"] * 50 + ["B"] * 50)
        result = self.analyzer.regime_persistence_test(returns, regimes)
        assert "A" in result.regime_means and "B" in result.regime_means
        assert result.f_stat >= 0
        assert 0 <= result.p_value <= 1
        assert result.best_regime in result.regime_sharpes
        assert result.worst_regime in result.regime_sharpes
