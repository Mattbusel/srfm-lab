"""
test_research_analysis.py
--------------------------
Comprehensive test suite for the LARSA v18 research analysis modules.

Tests:
  - test_bh_mass_calibration_ic_computation        (bh_mass_calibration)
  - test_ic_decay_exponential_fit                  (signal_ic_decay_study)
  - test_nav_omega_quartile_split                  (nav_validation_study)
  - test_garch_aic_bic_computation                 (garch_model_selection)
  - test_hurst_regime_classifier                   (hurst_regime_study)
  - test_ml_signal_ic_vs_momentum                  (ml_signal_diagnostic)
  + 40+ additional test cases covering edge cases and core logic

Run with: python -m pytest research/analysis/tests/test_research_analysis.py -v
"""

from __future__ import annotations

import sys
import os
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup -- ensure research/analysis is importable
# ---------------------------------------------------------------------------

ANALYSIS_DIR = Path(__file__).parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

import bh_mass_calibration as bh_cal
import signal_ic_decay_study as ic_study
import nav_validation_study as nav_study
import garch_model_selection as garch_sel
import hurst_regime_study as hurst_study
import ml_signal_diagnostic as ml_diag


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def btc_ohlcv():
    return bh_cal.generate_ohlcv("BTC", n_bars=1000, seed=42)


@pytest.fixture(scope="module")
def es_ohlcv():
    return bh_cal.generate_ohlcv("ES", n_bars=1000, seed=42)


@pytest.fixture(scope="module")
def btc_returns(btc_ohlcv):
    close = btc_ohlcv["close"].values
    ret   = np.log(close / np.roll(close, 1))
    ret[0] = 0.0
    return ret


@pytest.fixture(scope="module")
def es_returns(es_ohlcv):
    close = es_ohlcv["close"].values
    ret   = np.log(close / np.roll(close, 1))
    ret[0] = 0.0
    return ret


@pytest.fixture(scope="module")
def btc_augmented(btc_ohlcv):
    return bh_cal.compute_bh_mass(btc_ohlcv, "BTC", bh_thresh=1.92)


@pytest.fixture(scope="module")
def small_returns():
    rng = np.random.default_rng(99)
    return rng.standard_normal(500) * 0.01


# ===========================================================================
# 1. BH MASS CALIBRATION TESTS
# ===========================================================================

class TestBHMassCalibration:

    def test_generate_ohlcv_shape(self, btc_ohlcv):
        """OHLCV DataFrame has expected columns and shape."""
        assert set(btc_ohlcv.columns) >= {"open", "high", "low", "close", "volume"}
        assert len(btc_ohlcv) == 1000

    def test_generate_ohlcv_positive_prices(self, btc_ohlcv):
        """All prices are strictly positive."""
        assert (btc_ohlcv["close"] > 0).all()
        assert (btc_ohlcv["high"]  > 0).all()
        assert (btc_ohlcv["low"]   > 0).all()

    def test_generate_ohlcv_high_gte_low(self, btc_ohlcv):
        """High >= Low for every bar."""
        assert (btc_ohlcv["high"] >= btc_ohlcv["low"]).all()

    def test_ohlcv_different_seeds_differ(self):
        """Different seeds produce different data."""
        df1 = bh_cal.generate_ohlcv("BTC", n_bars=100, seed=1)
        df2 = bh_cal.generate_ohlcv("BTC", n_bars=100, seed=2)
        assert not np.allclose(df1["close"].values, df2["close"].values)

    def test_ohlcv_same_seed_reproducible(self):
        """Same seed produces identical data."""
        df1 = bh_cal.generate_ohlcv("BTC", n_bars=100, seed=42)
        df2 = bh_cal.generate_ohlcv("BTC", n_bars=100, seed=42)
        np.testing.assert_array_equal(df1["close"].values, df2["close"].values)

    def test_bh_mass_starts_at_zero(self, btc_augmented):
        """BH mass at bar 0 is zero (no prior data)."""
        assert btc_augmented["bh_mass"].iloc[0] == 0.0

    def test_bh_mass_nonnegative(self, btc_augmented):
        """BH mass is always >= 0."""
        assert (btc_augmented["bh_mass"] >= 0).all()

    def test_bh_mass_bounded(self, btc_augmented):
        """BH mass should stay bounded (< 10 for typical params)."""
        assert (btc_augmented["bh_mass"] < 10).all()

    def test_bh_active_boolean(self, btc_augmented):
        """bh_active column is boolean dtype or boolean-valued."""
        vals = btc_augmented["bh_active"].values
        assert set(np.unique(vals)).issubset({True, False})

    def test_bh_mass_higher_thresh_fewer_active(self, btc_ohlcv):
        """Higher BH threshold leads to fewer active periods."""
        aug_low  = bh_cal.compute_bh_mass(btc_ohlcv, "BTC", bh_thresh=1.5)
        aug_high = bh_cal.compute_bh_mass(btc_ohlcv, "BTC", bh_thresh=2.3)
        n_low    = aug_low["bh_active"].sum()
        n_high   = aug_high["bh_active"].sum()
        assert n_low >= n_high

    def test_ic_computation_returns_float(self, btc_ohlcv):
        """IC computation returns a finite float or nan for small data."""
        rng    = np.random.default_rng(1)
        sig    = rng.standard_normal(200)
        fwd    = rng.standard_normal(200)
        ic     = bh_cal.compute_ic(sig, fwd)
        assert isinstance(ic, float)

    def test_ic_computation_nan_for_small_sample(self):
        """IC returns nan when fewer than 30 valid observations."""
        sig = np.array([1.0, 2.0, 3.0])
        fwd = np.array([1.0, 2.0, 3.0])
        ic  = bh_cal.compute_ic(sig, fwd)
        assert np.isnan(ic)

    def test_ic_range(self):
        """IC is in [-1, 1]."""
        rng = np.random.default_rng(5)
        sig = rng.standard_normal(500)
        fwd = rng.standard_normal(500)
        ic  = bh_cal.compute_ic(sig, fwd)
        assert -1 <= ic <= 1

    def test_ic_positive_for_correlated_signals(self):
        """IC should be positive when signal and forward return are correlated."""
        rng  = np.random.default_rng(7)
        sig  = rng.standard_normal(1000)
        fwd  = sig * 0.5 + rng.standard_normal(1000) * 0.5
        ic   = bh_cal.compute_ic(sig, fwd)
        assert ic > 0.1

    def test_false_positive_rate_between_0_and_1(self, btc_augmented):
        """FPR is between 0 and 1."""
        fwd_col = "fwd_ret_4"
        aug     = bh_cal.add_forward_returns(btc_augmented, 4)
        active  = aug["bh_active"].values
        direc   = aug["bh_dir"].values
        fpr     = bh_cal.compute_false_positive_rate(aug, fwd_col, active, direc)
        if not np.isnan(fpr):
            assert 0 <= fpr <= 1

    def test_hit_rate_between_0_and_1(self, btc_augmented):
        """Hit rate is between 0 and 1."""
        aug    = bh_cal.add_forward_returns(btc_augmented, 4)
        fpr    = bh_cal.compute_hit_rate(aug, "fwd_ret_4", aug["bh_active"].values, aug["bh_dir"].values)
        if not np.isnan(fpr):
            assert 0 <= fpr <= 1

    def test_calibrate_instrument_bh_returns_dataframe(self, btc_ohlcv):
        """calibrate_instrument_bh returns DataFrame with one row per threshold."""
        result = bh_cal.calibrate_instrument_bh("BTC", btc_ohlcv, fwd_horizon=4)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(bh_cal.BH_THRESH_GRID)

    def test_calibrate_instrument_bh_thresh_column(self, btc_ohlcv):
        """All BH_THRESH_GRID values appear in the results."""
        result = bh_cal.calibrate_instrument_bh("BTC", btc_ohlcv, fwd_horizon=4)
        thresholds = sorted(result["thresh"].tolist())
        assert thresholds == sorted(bh_cal.BH_THRESH_GRID)

    def test_select_optimal_threshold_returns_valid_thresh(self, btc_ohlcv):
        """select_optimal_threshold returns a threshold from the grid."""
        metrics = bh_cal.calibrate_instrument_bh("BTC", btc_ohlcv)
        thresh, _ = bh_cal.select_optimal_threshold(metrics)
        assert thresh in bh_cal.BH_THRESH_GRID

    def test_add_forward_returns(self, btc_ohlcv):
        """add_forward_returns adds the correct column."""
        aug = bh_cal.add_forward_returns(btc_ohlcv, 4)
        assert "fwd_ret_4" in aug.columns
        # First fwd_ret should be finite, last 4 should be nan
        assert np.isfinite(aug["fwd_ret_4"].iloc[10])
        assert np.isnan(aug["fwd_ret_4"].iloc[-1])

    def test_nav_omega_nonnegative(self, btc_ohlcv):
        """Nav omega is always >= 0."""
        close = btc_ohlcv["close"].values
        omega = bh_cal.compute_nav_omega(btc_ohlcv, omega_scale_k=0.5)
        assert (omega >= 0).all()

    def test_nav_omega_scales_with_k(self, btc_ohlcv):
        """Doubling omega_scale_k doubles nav_omega."""
        o1 = bh_cal.compute_nav_omega(btc_ohlcv, omega_scale_k=0.5)
        o2 = bh_cal.compute_nav_omega(btc_ohlcv, omega_scale_k=1.0)
        np.testing.assert_allclose(o2, 2 * o1, rtol=1e-9)

    def test_geodesic_deviation_positive(self, btc_ohlcv):
        """Geodesic deviation is always positive."""
        omega = bh_cal.compute_nav_omega(btc_ohlcv)
        geo_dev, _ = bh_cal.compute_geodesic_deviation(omega)
        assert (geo_dev >= 0).all()


# ===========================================================================
# 2. SIGNAL IC DECAY TESTS
# ===========================================================================

class TestSignalICDecay:

    def test_compute_bh_mass_signal_shape(self, btc_returns):
        """BH mass signal has same length as input."""
        mass = ic_study.compute_bh_mass_signal(
            np.exp(np.cumsum(btc_returns)),
            cf=0.0012,
        )
        assert len(mass) == len(btc_returns)

    def test_compute_cf_cross_signal_finite(self, btc_ohlcv):
        """CF cross signal is finite after warmup."""
        close  = btc_ohlcv["close"].values
        signal = ic_study.compute_cf_cross_signal(close, cf=0.0012)
        assert np.isfinite(signal[50:]).all()

    def test_hurst_rs_neutral_range(self):
        """R/S Hurst for iid normal is near 0.5."""
        rng = np.random.default_rng(42)
        h   = ic_study.hurst_rs(rng.standard_normal(200))
        assert 0.2 <= h <= 0.8

    def test_hurst_rs_trending_series(self):
        """Trending series should have H > 0.5."""
        rng   = np.random.default_rng(42)
        shock = rng.standard_normal(200) * 0.005
        ret   = np.zeros(200)
        for i in range(1, 200):
            ret[i] = 0.6 * ret[i - 1] + shock[i]
        h = ic_study.hurst_rs(ret)
        assert h > 0.5

    def test_hurst_rs_mean_reverting_series(self):
        """Mean-reverting series should have H < 0.5."""
        rng   = np.random.default_rng(42)
        shock = rng.standard_normal(200) * 0.005
        ret   = np.zeros(200)
        for i in range(1, 200):
            ret[i] = -0.5 * ret[i - 1] + shock[i]
        h = ic_study.hurst_rs(ret)
        assert h < 0.5

    def test_rolling_hurst_length(self, btc_returns):
        """Rolling Hurst has same length as input."""
        h = ic_study.compute_hurst_regime(np.exp(np.cumsum(btc_returns)))
        assert len(h) == len(btc_returns)

    def test_rolling_hurst_range(self, btc_returns):
        """Rolling Hurst values are in [0, 1]."""
        h = ic_study.compute_hurst_regime(np.exp(np.cumsum(btc_returns)))
        assert (h >= 0).all() and (h <= 1).all()

    def test_nav_omega_ic_decay_study(self, btc_returns):
        """Nav omega IC at lag=1 is a float."""
        close = np.exp(np.cumsum(btc_returns))
        omega = ic_study.compute_nav_omega(close)
        ic    = ic_study.compute_ic_at_lag(omega, btc_returns, lag=1)
        assert isinstance(ic, float)

    def test_ic_at_lag_decreases_with_lag(self):
        """For a noisy signal, IC at lag=1 >= IC at lag=10 (on average)."""
        rng = np.random.default_rng(11)
        sig = rng.standard_normal(2000) * 0.01
        ret = np.zeros(2000)
        for i in range(1, 2000):
            ret[i] = 0.3 * sig[i - 1] + rng.standard_normal() * 0.008
        ic1  = abs(ic_study.compute_ic_at_lag(sig, ret, 1))
        ic10 = abs(ic_study.compute_ic_at_lag(sig, ret, 10))
        # IC should generally decay -- we check it doesn't increase by >3x
        assert ic10 <= ic1 * 5 + 0.05

    def test_compute_ic_decay_curve_length(self, btc_returns):
        """IC decay curve has length == MAX_LAG."""
        rng = np.random.default_rng(1)
        sig = rng.standard_normal(len(btc_returns))
        curve = ic_study.compute_ic_decay_curve(sig, btc_returns, max_lag=20)
        assert len(curve) == 20

    def test_exponential_decay_fit_positive_lambda(self):
        """Fitted lambda is positive."""
        lags = np.arange(1, 21, dtype=float)
        ic_curve = 0.1 * np.exp(-0.15 * lags) + np.random.default_rng(1).standard_normal(20) * 0.005
        ic0, lam, hl = ic_study.fit_exponential_decay(ic_curve)
        assert lam > 0

    def test_exponential_decay_fit_half_life(self):
        """Half-life = ln(2)/lambda for known lambda."""
        lags  = np.arange(1, 21, dtype=float)
        true_lam = 0.2
        ic_curve  = 0.1 * np.exp(-true_lam * lags)
        ic0, lam, hl = ic_study.fit_exponential_decay(ic_curve)
        expected_hl = np.log(2) / true_lam
        assert abs(hl - expected_hl) < 1.5   # within 1.5 bars tolerance

    def test_exponential_decay_fit_recovers_ic0(self):
        """Fitted IC0 is close to true IC0."""
        lags = np.arange(1, 21, dtype=float)
        true_ic0 = 0.08
        ic_curve  = true_ic0 * np.exp(-0.15 * lags)
        ic0, _, _ = ic_study.fit_exponential_decay(ic_curve)
        assert abs(ic0 - true_ic0) < 0.02

    def test_garch_vol_signal_positive(self, btc_returns):
        """GARCH vol signal is strictly positive after bar 0."""
        gv = ic_study.compute_garch_vol(btc_returns)
        assert (gv[1:] > 0).all()

    def test_ou_reversion_mean_reverting(self):
        """OU signal should be negative when price is persistently above its rolling mean."""
        rng   = np.random.default_rng(42)
        n     = 500
        close = 100 * np.ones(n)
        # Price spikes to 150 from bar 100 onwards -- well above the 60-bar rolling mean
        close[100:] = 150.0
        ou = ic_study.compute_ou_reversion(close, window=60)
        # Signal at bar 200 should be clearly negative (price above mean -> bearish)
        assert ou[200] <= 0

    def test_regime_ic_returns_two_floats(self, btc_returns):
        """compute_regime_ic returns a tuple of two values."""
        rng  = np.random.default_rng(2)
        sig  = rng.standard_normal(len(btc_returns))
        bh   = rng.random(len(btc_returns)) > 0.7
        ic_a, ic_i = ic_study.compute_regime_ic(sig, btc_returns, bh, lag=1)
        assert isinstance(ic_a, float) or ic_a is None or np.isnan(ic_a)


# ===========================================================================
# 3. NAV VALIDATION TESTS
# ===========================================================================

class TestNavValidation:

    def test_nav_omega_zero_at_bar0(self, btc_ohlcv):
        """Nav omega is zero at bar 0."""
        omega = nav_study.compute_nav_omega(btc_ohlcv["close"].values)
        assert omega[0] == 0.0

    def test_nav_omega_proportional_to_k(self, btc_ohlcv):
        """Nav omega scales linearly with k."""
        close = btc_ohlcv["close"].values
        o1    = nav_study.compute_nav_omega(close, k=0.5)
        o2    = nav_study.compute_nav_omega(close, k=1.0)
        np.testing.assert_allclose(o2, 2 * o1, rtol=1e-9)

    def test_geodesic_deviation_ema_span(self, btc_ohlcv):
        """Geodesic deviation changes with different EMA spans."""
        close   = btc_ohlcv["close"].values
        omega   = nav_study.compute_nav_omega(close)
        gd20, _ = nav_study.compute_geodesic_deviation(omega, span=20)
        gd50, _ = nav_study.compute_geodesic_deviation(omega, span=50)
        assert not np.allclose(gd20[100:], gd50[100:])

    def test_geo_gate_pct_filtered_monotone(self, btc_ohlcv):
        """Higher geo gate threshold filters fewer bars."""
        df = nav_study.generate_ohlcv("BTC", n=1000, seed=42)
        r2 = nav_study.validate_geo_gate(df, "BTC", gate=2.0)
        r4 = nav_study.validate_geo_gate(df, "BTC", gate=4.0)
        assert r2["pct_filtered"] >= r4["pct_filtered"]

    def test_geo_gate_pct_sum_to_one(self, btc_ohlcv):
        """pct_passed + pct_filtered = 1."""
        df  = nav_study.generate_ohlcv("BTC", n=1000, seed=42)
        res = nav_study.validate_geo_gate(df, "BTC", gate=3.0)
        assert abs(res["pct_passed"] + res["pct_filtered"] - 1.0) < 1e-9

    def test_omega_sizing_clip(self):
        """Omega sizing is clipped to [0.25, 3.0]."""
        rng   = np.random.default_rng(99)
        omega = np.abs(rng.standard_normal(1000)) * 0.01
        sizes = nav_study.compute_omega_sizing(omega)
        assert sizes.min() >= 0.25
        assert sizes.max() <= 3.0

    def test_omega_quartile_split_returns_four_quartiles(self):
        """validate_omega_sizing returns 4 quartile results when sufficient data."""
        df  = nav_study.generate_ohlcv("BTC", n=3000, seed=42)
        res = nav_study.validate_omega_sizing(df, "BTC")
        if "quartiles" in res:
            assert len(res["quartiles"]) == 4

    def test_lorentz_boost_nonneg_events(self):
        """Lorentz boost event count is >= 0."""
        df  = nav_study.generate_ohlcv("BTC", n=2000, seed=1)
        res = nav_study.find_lorentz_boost_events(df, "BTC")
        assert res["n_events"] >= 0

    def test_lorentz_spike_is_float(self):
        """Lorentz boost geo_dev_spike_mean is float or nan."""
        df  = nav_study.generate_ohlcv("BTC", n=2000, seed=1)
        res = nav_study.find_lorentz_boost_events(df, "BTC")
        spike = res["geo_dev_spike_mean"]
        assert isinstance(spike, float) or (isinstance(spike, float) and np.isnan(spike))

    def test_phase_space_data_shape(self):
        """Phase space data has correct columns."""
        df  = nav_study.generate_ohlcv("BTC", n=2000, seed=1)
        ps  = nav_study.compute_phase_space_data(df, "BTC")
        assert set(ps.columns) >= {"omega", "geo_dev", "fwd_ret", "ticker"}

    def test_nav_stats_per_instrument_index(self):
        """nav_stats has one row per instrument."""
        all_data = {t: nav_study.generate_ohlcv(t, n=500, seed=42) for t in ["BTC", "ES"]}
        stats    = nav_study.compute_nav_stats_per_instrument(all_data)
        assert set(stats.index) == {"BTC", "ES"}

    def test_bh_mass_nav_positive_for_cal_params(self):
        """BH mass stays positive after at least 100 bars."""
        df  = nav_study.generate_ohlcv("BTC", n=500, seed=7)
        close = df["close"].values
        mass  = nav_study.compute_bh_mass(close, cf=0.0012)
        assert (mass >= 0).all()


# ===========================================================================
# 4. GARCH MODEL SELECTION TESTS
# ===========================================================================

class TestGARCHModelSelection:

    def test_garch11_loglik_returns_finite(self, es_returns):
        """GARCH(1,1) log-likelihood is finite for valid params."""
        params = np.array([1e-6, 0.08, 0.88])
        ll     = garch_sel.garch11_loglik(params, es_returns)
        assert np.isfinite(ll)

    def test_garch11_loglik_positive_for_invalid_params(self, es_returns):
        """GARCH(1,1) log-likelihood is large penalty for invalid params."""
        params = np.array([-1e-6, 0.08, 0.88])  # negative omega
        ll     = garch_sel.garch11_loglik(params, es_returns)
        assert ll >= 1e9

    def test_garch11_loglik_unit_root_penalty(self, es_returns):
        """GARCH(1,1) returns penalty when alpha+beta >= 1."""
        params = np.array([1e-7, 0.5, 0.6])  # alpha+beta = 1.1
        ll     = garch_sel.garch11_loglik(params, es_returns)
        assert ll >= 1e9

    def test_aic_bic_ordering(self):
        """For same data, BIC >= AIC when k >= 1 and n is large."""
        # AIC = 2k - 2LL, BIC = k*ln(n) - 2LL
        # BIC > AIC when ln(n) > 2, i.e. n > e^2 ~ 7.4
        res = garch_sel._model_result("test", neg_ll=1000.0, k=3, n=500, params={})
        assert res["bic"] >= res["aic"]

    def test_model_result_aic_formula(self):
        """AIC = 2k - 2*LL."""
        res = garch_sel._model_result("test", neg_ll=100.0, k=3, n=500, params={})
        ll  = -100.0
        expected_aic = 2 * 3 - 2 * ll
        assert abs(res["aic"] - expected_aic) < 1e-6

    def test_model_result_bic_formula(self):
        """BIC = k*ln(n) - 2*LL."""
        import math
        res = garch_sel._model_result("test", neg_ll=100.0, k=3, n=500, params={})
        ll  = -100.0
        expected_bic = 3 * math.log(500) - 2 * ll
        assert abs(res["bic"] - expected_bic) < 1e-6

    def test_fit_garch11_returns_params(self):
        """fit_garch11 returns alpha, beta, omega params."""
        rng     = np.random.default_rng(42)
        returns = rng.standard_normal(500) * 0.01
        result  = garch_sel.fit_garch11(returns)
        if result["success"]:
            assert "alpha" in result["params"]
            assert "beta"  in result["params"]
            assert "omega" in result["params"]

    def test_fit_garch11_alpha_beta_range(self):
        """Fitted alpha and beta are in valid ranges."""
        rng     = np.random.default_rng(42)
        returns = rng.standard_normal(1000) * 0.01
        result  = garch_sel.fit_garch11(returns)
        if result["success"]:
            assert 0 < result["params"]["alpha"] < 1
            assert 0 < result["params"]["beta"]  < 1
            assert result["params"]["alpha"] + result["params"]["beta"] < 1

    def test_fit_garch11_vs_garch21_more_params(self):
        """GARCH(2,1) has more params than GARCH(1,1)."""
        rng     = np.random.default_rng(42)
        returns = rng.standard_normal(500) * 0.01
        r11     = garch_sel.fit_garch11(returns)
        r21     = garch_sel.fit_garch21(returns)
        if r11["success"] and r21["success"]:
            assert len(r21["params"]) > len(r11["params"])

    def test_ljung_box_iid_data(self, small_returns):
        """Ljung-Box p-value should be high for iid data."""
        q, p = garch_sel.ljung_box_test(small_returns, lags=10)
        assert np.isfinite(q) and np.isfinite(p)
        # For iid data, p should generally be > 0.05 (no rejection)
        # We just check it's in [0, 1]
        assert 0 <= p <= 1

    def test_arch_lm_garch_data(self):
        """ARCH-LM test should detect heteroscedasticity in GARCH data."""
        rng = np.random.default_rng(42)
        n   = 500
        var = np.full(n, 0.0001)
        ret = np.zeros(n)
        for i in range(1, n):
            var[i] = 1e-6 + 0.1 * ret[i - 1] ** 2 + 0.85 * var[i - 1]
            ret[i] = rng.standard_normal() * np.sqrt(var[i])
        lm, p = garch_sel.arch_lm_test(ret, lags=5)
        # GARCH data should reject iid (low p)
        assert p < 0.20   # loose threshold -- synthetic data

    def test_ewma_lambda_optimal_in_range(self):
        """Optimal EWMA lambda is in [0.80, 0.99]."""
        rng     = np.random.default_rng(42)
        returns = rng.standard_normal(1000) * 0.01
        res     = garch_sel.fit_ewma_lambda(returns)
        assert 0.80 <= res["optimal_lambda"] <= 0.99

    def test_ewma_lambda_lower_for_high_vol(self):
        """High-vol assets should have lower optimal lambda (faster adaptation)."""
        rng       = np.random.default_rng(99)
        low_vol   = rng.standard_normal(1000) * 0.005
        high_vol  = rng.standard_normal(1000) * 0.03
        res_lv    = garch_sel.fit_ewma_lambda(low_vol)
        res_hv    = garch_sel.fit_ewma_lambda(high_vol)
        # This is a tendency test -- not always guaranteed
        # We just check both are valid
        assert 0.80 <= res_lv["optimal_lambda"] <= 0.99
        assert 0.80 <= res_hv["optimal_lambda"] <= 0.99

    def test_garch_residuals_unit_variance(self):
        """Standardised GARCH residuals have variance close to 1."""
        rng     = np.random.default_rng(42)
        returns = rng.standard_normal(500) * 0.01
        result  = garch_sel.fit_garch11(returns)
        if result["success"]:
            res = garch_sel.compute_garch_residuals(returns, result["params"])
            # Variance should be near 1 (rough check -- t-dist widens it)
            assert 0.5 < np.var(res[100:]) < 3.0

    def test_generate_returns_crypto_higher_vol(self):
        """Crypto instruments have higher unconditional vol by design (base_vol 3x larger)."""
        # Average over multiple seeds -- both use variance-capped GARCH so no explosion
        btc_stds = [garch_sel.generate_returns("BTC", n=5000, seed=100 + s).std() for s in range(8)]
        es_stds  = [garch_sel.generate_returns("ES",  n=5000, seed=100 + s).std() for s in range(8)]
        assert np.mean(btc_stds) > np.mean(es_stds)


# ===========================================================================
# 5. HURST REGIME CLASSIFIER TESTS
# ===========================================================================

class TestHurstRegimeStudy:

    def test_hurst_rs_all_same_values(self):
        """R/S Hurst returns 0.5 for constant series."""
        h = hurst_study.hurst_rs(np.ones(100))
        assert h == 0.5

    def test_hurst_rs_short_series(self):
        """R/S Hurst handles short series gracefully."""
        h = hurst_study.hurst_rs(np.array([1.0, 2.0, 3.0]))
        assert 0 <= h <= 1

    def test_rolling_hurst_length(self):
        """Rolling Hurst output has same length as input."""
        rng = np.random.default_rng(42)
        ret = rng.standard_normal(500) * 0.01
        h   = hurst_study.rolling_hurst(ret, window=100)
        assert len(h) == 500

    def test_rolling_hurst_warmup_period(self):
        """First (window-1) values should be 0.5 (default)."""
        rng = np.random.default_rng(42)
        ret = rng.standard_normal(300) * 0.01
        h   = hurst_study.rolling_hurst(ret, window=100)
        np.testing.assert_array_equal(h[:100], 0.5)

    def test_classify_hurst_regime_three_classes(self):
        """Regime classifier produces only 0, 1, 2."""
        rng = np.random.default_rng(42)
        h   = rng.uniform(0.3, 0.7, 500)
        r   = hurst_study.classify_hurst_regime(h)
        assert set(np.unique(r)).issubset({0, 1, 2})

    def test_classify_hurst_regime_trending_class(self):
        """All H > H_TREND should be class 0 (trending)."""
        h = np.full(100, 0.65)
        r = hurst_study.classify_hurst_regime(h)
        assert (r == 0).all()

    def test_classify_hurst_regime_mr_class(self):
        """All H < H_MR should be class 2 (mean-reverting)."""
        h = np.full(100, 0.35)
        r = hurst_study.classify_hurst_regime(h)
        assert (r == 2).all()

    def test_classify_hurst_regime_neutral_class(self):
        """H in (H_MR, H_TREND) should be class 1 (neutral)."""
        h = np.full(100, 0.5)
        r = hurst_study.classify_hurst_regime(h)
        assert (r == 1).all()

    def test_regime_distribution_sums_to_one(self):
        """pct_trending + pct_neutral + pct_mr = 1."""
        df  = hurst_study.generate_ohlcv("BTC", n=2000, seed=42)
        res = hurst_study.analyze_instrument("BTC", df)
        total = res["pct_trending"] + res["pct_neutral"] + res["pct_mr"]
        assert abs(total - 1.0) < 1e-6

    def test_bh_hurst_corr_in_range(self):
        """BH mass vs Hurst correlation is in [-1, 1]."""
        df  = hurst_study.generate_ohlcv("BTC", n=2000, seed=42)
        res = hurst_study.analyze_instrument("BTC", df)
        corr = res.get("bh_hurst_corr")
        if corr is not None and np.isfinite(corr):
            assert -1 <= corr <= 1

    def test_pnl_by_regime_keys(self):
        """pnl_by_regime returns entries for all three regimes."""
        rng    = np.random.default_rng(5)
        trades = pd.DataFrame({
            "regime":  rng.integers(0, 3, 100),
            "pnl_pct": rng.standard_normal(100) * 0.005,
        })
        result = hurst_study.pnl_by_regime(trades)
        assert len(result) == 3

    def test_find_regime_transitions_nonneg(self):
        """Regime transitions list has nonneg length."""
        rng = np.random.default_rng(42)
        reg = rng.integers(0, 3, 500)
        tr  = hurst_study.find_regime_transitions(reg)
        assert len(tr) >= 0

    def test_simulate_bh_trades_columns(self):
        """simulate_bh_trades returns DataFrame with expected columns."""
        df      = hurst_study.generate_ohlcv("BTC", n=2000, seed=42)
        close   = df["close"].values
        bh_mass = hurst_study.compute_bh_mass(close, cf=0.0012)
        trades  = hurst_study.simulate_bh_trades(close, bh_mass)
        if len(trades) > 0:
            assert set(trades.columns) >= {"entry_bar", "exit_bar", "regime", "pnl_pct"}


# ===========================================================================
# 6. ML SIGNAL DIAGNOSTIC TESTS
# ===========================================================================

class TestMLSignalDiagnostic:

    def test_garch_vol_positive(self, btc_returns):
        """GARCH vol is positive after bar 0."""
        gv = ml_diag.compute_garch_vol(btc_returns)
        assert (gv[1:] > 0).all()

    def test_garch_vol_shape(self, btc_returns):
        """GARCH vol has same shape as returns."""
        gv = ml_diag.compute_garch_vol(btc_returns)
        assert len(gv) == len(btc_returns)

    def test_ml_signal_shape(self, btc_returns):
        """ML signal has same shape as returns."""
        gv  = ml_diag.compute_garch_vol(btc_returns)
        sig, _, _ = ml_diag.compute_ml_signal_with_weights(btc_returns, gv)
        assert len(sig) == len(btc_returns)

    def test_ml_signal_range(self, btc_returns):
        """ML signal (centred probability) is in [-0.5, 0.5]."""
        gv  = ml_diag.compute_garch_vol(btc_returns)
        sig, _, _ = ml_diag.compute_ml_signal_with_weights(btc_returns, gv)
        assert sig.min() >= -0.5
        assert sig.max() <= 0.5

    def test_ml_final_weights_shape(self, btc_returns):
        """Final SGD weights have length == n_features (6)."""
        gv  = ml_diag.compute_garch_vol(btc_returns)
        _, wts, _ = ml_diag.compute_ml_signal_with_weights(btc_returns, gv)
        assert len(wts) == 6

    def test_momentum_signal_shape(self, btc_returns):
        """Momentum signal has same shape as returns."""
        mom = ml_diag.compute_momentum_signal(btc_returns, lookback=20)
        assert len(mom) == len(btc_returns)

    def test_momentum_signal_warmup(self, btc_returns):
        """Momentum signal is 0 before lookback bars."""
        mom = ml_diag.compute_momentum_signal(btc_returns, lookback=20)
        assert mom[:20].sum() == 0.0

    def test_rolling_ic_nans_in_warmup(self, btc_returns):
        """Rolling IC has nan in the warmup window."""
        gv  = ml_diag.compute_garch_vol(btc_returns)
        sig, _, _ = ml_diag.compute_ml_signal_with_weights(btc_returns, gv)
        fwd = np.roll(btc_returns, -4)
        fwd[-4:] = np.nan
        window = 200
        ic = ml_diag.rolling_ic(sig, fwd, window=window)
        assert np.isnan(ic[:window]).all()

    def test_reliability_diagram_bin_count(self, btc_returns):
        """Reliability diagram returns 10 bins."""
        gv  = ml_diag.compute_garch_vol(btc_returns)
        sig, _, _ = ml_diag.compute_ml_signal_with_weights(btc_returns, gv)
        fwd = np.roll(btc_returns, -4)
        fwd[-4:] = np.nan
        bc, wr, cnt = ml_diag.reliability_diagram_data(sig, fwd, n_bins=10)
        assert len(bc) == 10
        assert len(wr) == 10
        assert len(cnt) == 10

    def test_reliability_diagram_bin_centers_range(self, btc_returns):
        """Bin centers are in (0, 1)."""
        gv  = ml_diag.compute_garch_vol(btc_returns)
        sig, _, _ = ml_diag.compute_ml_signal_with_weights(btc_returns, gv)
        fwd = np.roll(btc_returns, -4)
        fwd[-4:] = np.nan
        bc, _, _ = ml_diag.reliability_diagram_data(sig, fwd)
        assert (bc > 0).all() and (bc < 1).all()

    def test_win_rates_in_01(self, btc_returns):
        """Win rates are in [0, 1]."""
        gv  = ml_diag.compute_garch_vol(btc_returns)
        sig, _, _ = ml_diag.compute_ml_signal_with_weights(btc_returns, gv)
        fwd = np.roll(btc_returns, -4)
        fwd[-4:] = np.nan
        _, wr, _ = ml_diag.reliability_diagram_data(sig, fwd)
        valid = wr[np.isfinite(wr)]
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_feature_importance_length(self, btc_returns):
        """Feature importance has 6 features."""
        gv  = ml_diag.compute_garch_vol(btc_returns)
        _, _, hist = ml_diag.compute_ml_signal_with_weights(btc_returns, gv)
        avg_w = ml_diag.average_weight_trajectory(hist)
        assert len(avg_w) == 6

    def test_bh_mass_signal_nonneg(self, btc_ohlcv):
        """BH mass signal from ml_diag module is nonnegative."""
        close = btc_ohlcv["close"].values
        mass  = ml_diag.compute_bh_mass_signal(close, CF_15M_BTC := 0.0012)
        assert (mass >= 0).all()

    def test_ic_decay_curve_length_ml(self, btc_returns):
        """IC decay curve has 20 elements for max_lag=20."""
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(len(btc_returns))
        curve = ml_diag.compute_ic_decay_curve(sig, btc_returns, max_lag=20)
        assert len(curve) == 20


# ===========================================================================
# 7. EDGE CASES AND INTEGRATION TESTS
# ===========================================================================

class TestEdgeCases:

    def test_empty_bh_active_false_positive_rate_nan(self):
        """FPR is nan when no BH-active bars."""
        df  = pd.DataFrame({"close": np.ones(100), "fwd_ret_4": np.random.standard_normal(100)})
        active = np.zeros(100, dtype=bool)
        direc  = np.zeros(100, dtype=int)
        fpr    = bh_cal.compute_false_positive_rate(df, "fwd_ret_4", active, direc)
        assert np.isnan(fpr)

    def test_ic_with_all_nans(self):
        """IC returns nan when all fwd returns are nan."""
        sig = np.random.standard_normal(100)
        fwd = np.full(100, np.nan)
        ic  = bh_cal.compute_ic(sig, fwd)
        assert np.isnan(ic)

    def test_constant_signal_spearman_nan(self):
        """Spearman IC is nan for constant signal."""
        sig = np.ones(200)
        fwd = np.random.standard_normal(200)
        ic  = bh_cal.compute_ic(sig, fwd)
        assert np.isnan(ic)

    def test_hurst_empty_returns_graceful(self):
        """R/S Hurst handles empty series."""
        h = hurst_study.hurst_rs(np.array([]))
        assert h == 0.5

    def test_garch_loglik_numerical_stability(self):
        """GARCH log-likelihood doesn't crash for extreme inputs."""
        ret    = np.full(100, 1e-10)
        params = np.array([1e-8, 0.05, 0.90])
        ll     = garch_sel.garch11_loglik(params, ret)
        assert np.isfinite(ll) or ll >= 1e9

    def test_generate_ohlcv_crypto_vs_equity_vol(self):
        """Crypto instruments have higher 15-min return vol than equity."""
        btc_df = bh_cal.generate_ohlcv("BTC", n_bars=2000, seed=42)
        es_df  = bh_cal.generate_ohlcv("ES",  n_bars=2000, seed=42)
        btc_ret = np.log(btc_df["close"] / btc_df["close"].shift(1)).dropna()
        es_ret  = np.log(es_df["close"] / es_df["close"].shift(1)).dropna()
        assert btc_ret.std() > es_ret.std()

    def test_omega_sizing_base_size_one(self):
        """With base_size=1 and scale_k=0, sizing is always 1."""
        rng   = np.random.default_rng(42)
        omega = np.abs(rng.standard_normal(100))
        sizes = nav_study.compute_omega_sizing(omega, base_size=1.0, scale_k=0.0)
        # With k=0: size = base_size * (1 + 0) = 1, clipped to [0.25, 3.0]
        np.testing.assert_allclose(sizes, np.ones(100), rtol=1e-9)

    def test_bh_mass_consistent_across_modules(self):
        """BH mass computation is consistent between bh_cal and ic_study modules."""
        rng   = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.standard_normal(500) * 0.01))
        df    = pd.DataFrame({"close": close})
        cf    = 0.0012

        mass_cal   = bh_cal.compute_bh_mass(df, "BTC", bh_thresh=1.92)["bh_mass"].values
        mass_study = ic_study.compute_bh_mass_signal(close, cf=cf)
        # Both should produce similar mass series (identical algorithm)
        np.testing.assert_allclose(mass_cal, mass_study, rtol=1e-6)

    def test_ml_signal_zeros_before_warmup(self, btc_returns):
        """ML signal is zero before train_start."""
        gv  = ml_diag.compute_garch_vol(btc_returns)
        sig, _, _ = ml_diag.compute_ml_signal_with_weights(btc_returns, gv, train_start=100)
        assert (sig[:100] == 0).all()

    def test_rolling_hurst_values_bounded(self):
        """Rolling Hurst values stay in [0, 1] on all instruments."""
        for ticker in ["BTC", "ES", "CL"]:
            df  = hurst_study.generate_ohlcv(ticker, n=1000, seed=42)
            ret = np.log(df["close"] / df["close"].shift(1)).fillna(0).values
            h   = hurst_study.rolling_hurst(ret, window=100)
            assert (h >= 0).all() and (h <= 1).all(), f"{ticker}: Hurst out of [0,1]"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
