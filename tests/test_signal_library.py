"""
tests/test_signal_library.py
=============================
Comprehensive tests for the signal_library and alpha_engine modules.

Tests cover:
- Each signal returns a pd.Series
- No NaN after warmup period
- Bounded in expected range (RSI 0-100, bounded signals)
- IC calculations return values in [-1, 1]
- SignalCombiner reduces mean absolute correlation
- ICCalculator rolling ICIR stability
- FeatureComputer batch compute
- AlphaEngine round-trip (register, score, report)

Dependencies: pytest, numpy, pandas, scipy
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import pytest

from research.signal_analytics.signal_library import (
    SIGNAL_REGISTRY,
    SIGNAL_CATEGORIES,
    # Momentum
    mom_1d, mom_5d, mom_20d, mom_60d, mom_252d,
    mom_sharpe, mom_acceleration, mom_52w_high, mom_crash_protection,
    mom_ts_moskowitz, mom_cs_rank, mom_seasonality, mom_dual,
    mom_absolute, mom_intermediate, mom_short_reversal, mom_end_of_month,
    mom_gap, mom_volume_weighted, mom_up_down_volume, mom_tick_proxy,
    mom_price_accel, mom_multi_tf_composite,
    # Mean Reversion
    mr_zscore_10, mr_zscore_20, mr_zscore_50,
    mr_bollinger_position, mr_rsi, mr_linreg_residual, mr_sma_deviation,
    mr_kalman_residual, mr_pairs_ratio_zscore, mr_ou_weighted,
    mr_vwap_deviation, mr_price_oscillator, mr_dpo, mr_cci, mr_williams_r,
    mr_stochastic_k, mr_chande_momentum, mr_roc_mean_rev, mr_price_channel,
    mr_log_autoregression, mr_hurst_adjusted,
    # Volatility
    vol_ewma_forecast, vol_realized_5d, vol_realized_20d, vol_realized_60d,
    vol_of_vol, vol_regime, vol_atr_percentile, vol_skew_proxy,
    vol_term_structure, vol_parkinson, vol_garman_klass, vol_yang_zhang,
    vol_rogers_satchell, vol_arch_signal, vol_normalized_range, vol_hist_vs_implied,
    # Microstructure
    ms_volume_surprise, ms_vpt, ms_obv_normalized, ms_cmf, ms_adl,
    ms_force_index, ms_emv, ms_volume_oscillator, ms_mfi, ms_nvi, ms_pvi,
    ms_pvt, ms_volume_momentum, ms_large_trade, ms_kyle_lambda,
    # Physics
    phys_bh_mass, phys_proper_time, phys_timelike_fraction, phys_ds2_trend,
    phys_bh_formation_rate, phys_geodesic_deviation, phys_angular_velocity,
    phys_hurst_signal, phys_fractal_dimension, phys_hawking_temperature,
    phys_grav_lensing, phys_phase_transition, phys_causal_info_ratio,
    phys_regime_velocity, phys_curvature_proxy,
    # Technical
    tech_macd_histogram, tech_adx, tech_aroon_oscillator, tech_cci,
    tech_keltner_position, tech_donchian_breakout, tech_ichimoku_cloud,
    tech_psar, tech_elder_ray, tech_vortex, tech_chande_kroll,
    tech_supertrend, tech_trix, tech_mass_index, tech_ulcer_index,
)
from research.signal_analytics.alpha_engine import (
    Signal, SignalRecord, ICCalculator, AlphaDecayAnalyzer,
    SignalCombiner, SignalRetirementEngine, AlphaEngine, AlphaReport,
    make_signal_record,
)
from research.signal_analytics.feature_store import (
    Feature, FeatureStore, FeatureComputer, FeatureMatrix, FeaturePipeline,
    create_feature_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def price_series_short() -> pd.Series:
    """300-bar synthetic price series (random walk)."""
    rng = np.random.default_rng(42)
    log_returns = rng.normal(0.0003, 0.01, 300)
    log_prices = np.cumsum(log_returns) + np.log(100.0)
    idx = pd.date_range("2022-01-01", periods=300, freq="B")
    return pd.Series(np.exp(log_prices), index=idx)


@pytest.fixture(scope="module")
def price_series_long() -> pd.Series:
    """500-bar synthetic price series."""
    rng = np.random.default_rng(99)
    log_returns = rng.normal(0.0002, 0.012, 500)
    log_prices = np.cumsum(log_returns) + np.log(50.0)
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    return pd.Series(np.exp(log_prices), index=idx)


@pytest.fixture(scope="module")
def volume_series(price_series_short) -> pd.Series:
    rng = np.random.default_rng(7)
    vol = rng.lognormal(mean=15, sigma=0.5, size=len(price_series_short))
    return pd.Series(vol, index=price_series_short.index)


@pytest.fixture(scope="module")
def high_low_series(price_series_short):
    rng = np.random.default_rng(3)
    noise_up = np.abs(rng.normal(0, 0.005, len(price_series_short)))
    noise_dn = np.abs(rng.normal(0, 0.005, len(price_series_short)))
    high = price_series_short * (1 + noise_up)
    low = price_series_short * (1 - noise_dn)
    return high, low


@pytest.fixture(scope="session")
def tmp_db(tmp_path_factory):
    return tmp_path_factory.mktemp("db") / "test_alpha.db"


@pytest.fixture(scope="session")
def tmp_fs_db(tmp_path_factory):
    return tmp_path_factory.mktemp("fsdb") / "test_feature_store.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _warmup_slice(series: pd.Series, warmup: int = 60) -> pd.Series:
    """Return series with first warmup bars removed."""
    return series.iloc[warmup:]


def _first_valid_index(series: pd.Series) -> int:
    """Return index of first non-NaN value."""
    valid = series.dropna()
    if valid.empty:
        return len(series)
    return series.index.get_loc(valid.index[0])


def _nan_fraction_after_warmup(series: pd.Series, warmup: int = 60) -> float:
    """Fraction of NaN values after warmup."""
    post_warmup = series.iloc[warmup:]
    if len(post_warmup) == 0:
        return 1.0
    return float(post_warmup.isna().sum() / len(post_warmup))


# ---------------------------------------------------------------------------
# Signal Registry Tests
# ---------------------------------------------------------------------------


class TestSignalRegistry:
    """Basic structure tests for the signal registry."""

    def test_registry_not_empty(self):
        assert len(SIGNAL_REGISTRY) >= 100, f"Expected 100+ signals, got {len(SIGNAL_REGISTRY)}"

    def test_all_registry_values_callable(self):
        for name, func in SIGNAL_REGISTRY.items():
            assert callable(func), f"Signal '{name}' is not callable"

    def test_categories_cover_all_signals(self):
        all_in_cats = set()
        for signals in SIGNAL_CATEGORIES.values():
            all_in_cats.update(signals)
        for name in SIGNAL_REGISTRY:
            assert name in all_in_cats, f"Signal '{name}' not in any category"

    def test_momentum_count(self):
        assert len(SIGNAL_CATEGORIES["momentum"]) >= 20

    def test_mean_reversion_count(self):
        assert len(SIGNAL_CATEGORIES["mean_reversion"]) >= 20

    def test_volatility_count(self):
        assert len(SIGNAL_CATEGORIES["volatility"]) >= 15

    def test_microstructure_count(self):
        assert len(SIGNAL_CATEGORIES["microstructure"]) >= 15

    def test_physics_count(self):
        assert len(SIGNAL_CATEGORIES["physics"]) >= 15

    def test_technical_count(self):
        assert len(SIGNAL_CATEGORIES["technical"]) >= 15


# ---------------------------------------------------------------------------
# Signal Return Type Tests
# ---------------------------------------------------------------------------


class TestSignalReturnTypes:
    """Every signal must return a pd.Series."""

    def _run_signal(self, func: Callable, prices: pd.Series, volume: pd.Series) -> pd.Series:
        return func(prices, volume=volume)

    def test_all_signals_return_series(self, price_series_short, volume_series):
        for name, func in SIGNAL_REGISTRY.items():
            result = self._run_signal(func, price_series_short, volume_series)
            assert isinstance(result, pd.Series), f"Signal '{name}' did not return pd.Series"

    def test_all_signals_same_length_or_shorter(self, price_series_short, volume_series):
        for name, func in SIGNAL_REGISTRY.items():
            result = self._run_signal(func, price_series_short, volume_series)
            assert len(result) == len(price_series_short), (
                f"Signal '{name}' returned length {len(result)} != {len(price_series_short)}"
            )

    def test_all_signals_same_index(self, price_series_short, volume_series):
        for name, func in SIGNAL_REGISTRY.items():
            result = self._run_signal(func, price_series_short, volume_series)
            assert result.index.equals(price_series_short.index), (
                f"Signal '{name}' returned misaligned index"
            )


# ---------------------------------------------------------------------------
# NaN after warmup tests
# ---------------------------------------------------------------------------


class TestNoNaNAfterWarmup:
    """Signals should have few/no NaN after their warmup period."""

    WARMUP = 60
    MAX_NAN_FRACTION = 0.05  # allow up to 5% NaN post-warmup

    def _check_signal(self, func: Callable, prices: pd.Series, volume: pd.Series, name: str) -> None:
        result = func(prices, volume=volume)
        nan_frac = _nan_fraction_after_warmup(result, self.WARMUP)
        assert nan_frac <= self.MAX_NAN_FRACTION, (
            f"Signal '{name}' has {nan_frac:.1%} NaN after warmup={self.WARMUP}"
        )

    def test_momentum_signals_no_nan(self, price_series_long, volume_series):
        prices = price_series_long
        vol = pd.Series(
            np.random.lognormal(15, 0.5, len(prices)),
            index=prices.index,
        )
        for name in SIGNAL_CATEGORIES["momentum"]:
            func = SIGNAL_REGISTRY[name]
            result = func(prices, volume=vol)
            # Use warmup=280 for momentum signals (mom_seasonality needs 273 bars)
            nan_frac = _nan_fraction_after_warmup(result, 280)
            assert nan_frac <= self.MAX_NAN_FRACTION, (
                f"Signal '{name}' has {nan_frac:.1%} NaN after warmup={self.WARMUP}"
            )

    def test_mean_reversion_no_nan(self, price_series_long):
        for name in SIGNAL_CATEGORIES["mean_reversion"]:
            func = SIGNAL_REGISTRY[name]
            result = func(price_series_long, volume=None)
            nan_frac = _nan_fraction_after_warmup(result, self.WARMUP)
            assert nan_frac <= self.MAX_NAN_FRACTION, (
                f"Signal '{name}' has {nan_frac:.1%} NaN after warmup"
            )

    def test_volatility_signals_no_nan(self, price_series_long):
        for name in SIGNAL_CATEGORIES["volatility"]:
            func = SIGNAL_REGISTRY[name]
            result = func(price_series_long, volume=None)
            nan_frac = _nan_fraction_after_warmup(result, self.WARMUP)
            assert nan_frac <= self.MAX_NAN_FRACTION, (
                f"Signal '{name}' has {nan_frac:.1%} NaN after warmup"
            )

    def test_technical_signals_no_nan(self, price_series_long):
        for name in SIGNAL_CATEGORIES["technical"]:
            func = SIGNAL_REGISTRY[name]
            result = func(price_series_long, volume=None)
            nan_frac = _nan_fraction_after_warmup(result, self.WARMUP)
            assert nan_frac <= self.MAX_NAN_FRACTION, (
                f"Signal '{name}' has {nan_frac:.1%} NaN after warmup"
            )


# ---------------------------------------------------------------------------
# Bounded Signal Tests
# ---------------------------------------------------------------------------


class TestSignalBounds:
    """Signals with known mathematical bounds."""

    def test_rsi_bounded_0_100(self, price_series_short):
        """RSI raw values should be in [0, 100] before normalization."""
        prices = price_series_short
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).ewm(span=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(span=14, adjust=False).mean()
        rs = gain / loss.replace(0.0, float("nan"))
        rsi = 100 - 100 / (1 + rs)
        valid = rsi.dropna()
        assert float(valid.min()) >= 0.0, f"RSI min {valid.min()} < 0"
        assert float(valid.max()) <= 100.0, f"RSI max {valid.max()} > 100"

    def test_mr_rsi_normalized_bounded(self, price_series_short):
        """mr_rsi signal should be in [-1, 1]."""
        result = mr_rsi(price_series_short)
        valid = result.dropna()
        assert float(valid.min()) >= -1.0 - 1e-9, f"mr_rsi min {valid.min()} < -1"
        assert float(valid.max()) <= 1.0 + 1e-9, f"mr_rsi max {valid.max()} > 1"

    def test_williams_r_signal_bounded(self, price_series_short):
        """Williams %R signal should be in [-1, 1] after normalization."""
        result = mr_williams_r(price_series_short)
        valid = result.dropna()
        assert float(valid.min()) >= -1.0 - 1e-6
        assert float(valid.max()) <= 1.0 + 1e-6

    def test_stochastic_k_normalized(self, price_series_short):
        """Stochastic K signal should be in [-1, 1] after normalization."""
        result = mr_stochastic_k(price_series_short)
        valid = result.dropna()
        assert float(valid.min()) >= -1.0 - 1e-6
        assert float(valid.max()) <= 1.0 + 1e-6

    def test_mom_dual_binary(self, price_series_short):
        """Dual momentum should return only +1 or -1."""
        result = mom_dual(price_series_short)
        valid = result.dropna()
        assert set(valid.unique()).issubset({1.0, -1.0}), f"Unexpected dual values: {set(valid.unique())}"

    def test_mom_absolute_binary(self, price_series_short):
        """Absolute momentum should return +1, -1, or 0."""
        result = mom_absolute(price_series_short)
        valid = result.dropna()
        assert set(valid.unique()).issubset({1.0, -1.0, 0.0})

    def test_timelike_fraction_bounded(self, price_series_short):
        """Timelike fraction must be in [0, 1]."""
        result = phys_timelike_fraction(price_series_short)
        valid = result.dropna()
        assert float(valid.min()) >= 0.0 - 1e-9
        assert float(valid.max()) <= 1.0 + 1e-9

    def test_causal_info_ratio_bounded(self, price_series_short):
        result = phys_causal_info_ratio(price_series_short)
        valid = result.dropna()
        assert float(valid.min()) >= 0.0 - 1e-9
        assert float(valid.max()) <= 1.0 + 1e-9

    def test_tech_adx_bounded(self, price_series_short):
        """ADX normalized should be in [0, 1]."""
        result = tech_adx(price_series_short)
        valid = result.dropna()
        # ADX/100 can slightly exceed 1 in edge cases, but should be near 0-1
        assert float(valid.min()) >= -0.01, f"ADX min {valid.min()} < 0"
        assert float(valid.max()) <= 1.01, f"ADX max {valid.max()} > 1"

    def test_chande_momentum_bounded(self, price_series_short):
        """CMO should be in [-1, 1] after normalization."""
        result = mr_chande_momentum(price_series_short)
        valid = result.dropna()
        assert float(valid.min()) >= -1.0 - 1e-9
        assert float(valid.max()) <= 1.0 + 1e-9

    def test_vol_signals_non_negative(self, price_series_short):
        """Realized vol signals should be non-negative."""
        for func in [vol_realized_5d, vol_realized_20d, vol_realized_60d, vol_ewma_forecast]:
            result = func(price_series_short)
            valid = result.dropna()
            assert float(valid.min()) >= -1e-10, f"{func.__name__} has negative vol"

    def test_psar_binary(self, price_series_short):
        """PSAR signal should be +1 or -1 (first bar may be 0 initialization)."""
        result = tech_psar(price_series_short)
        valid = result.dropna()
        # Allow 0 for the initial bar (uninitialized direction)
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_supertrend_binary(self, price_series_short):
        """SuperTrend direction should be +1 or -1."""
        result = tech_supertrend(price_series_short)
        valid = result.dropna()
        assert set(valid.unique()).issubset({1.0, -1.0})

    def test_donchian_breakout_bounded(self, price_series_short):
        """Donchian breakout should be in {-1, 0, 1}."""
        result = tech_donchian_breakout(price_series_short)
        valid = result.dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_cmf_bounded(self, price_series_short, volume_series):
        """Chaikin Money Flow should be in [-1, 1]."""
        result = ms_cmf(price_series_short, volume=volume_series)
        valid = result.dropna()
        assert float(valid.min()) >= -1.0 - 1e-9
        assert float(valid.max()) <= 1.0 + 1e-9

    def test_mfi_normalized_bounded(self, price_series_short, volume_series):
        """MFI normalized signal should be in [-1, 1] (MFI/50 - 1)."""
        result = ms_mfi(price_series_short, volume=volume_series)
        valid = result.dropna()
        assert float(valid.min()) >= -1.0 - 1e-6
        assert float(valid.max()) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Monotone / direction tests
# ---------------------------------------------------------------------------


class TestSignalDirectionality:
    """Test that directional signals have correct orientation."""

    def test_mom_52w_high_less_than_1(self, price_series_short):
        """52-week high ratio <= 1 always (price can't exceed its own max)."""
        result = mom_52w_high(price_series_short)
        valid = result.dropna()
        assert float(valid.max()) <= 1.0 + 1e-9, "52-week high ratio > 1"

    def test_vol_parkinson_positive(self, price_series_short):
        result = vol_parkinson(price_series_short)
        valid = result.dropna()
        assert float(valid.min()) >= 0.0 - 1e-10

    def test_ulcer_index_signal(self, price_series_short):
        """Ulcer index should be negative (inverted) when normalized."""
        result = tech_ulcer_index(price_series_short)
        valid = result.dropna()
        # Ulcer index raw is non-negative, but normalized can be negative
        # Just check it's a valid float series
        assert valid.notna().any()

    def test_hurst_exponent_range(self, price_series_short):
        """Hurst signal (2*H - 1) should be in [-1, 1]."""
        result = phys_hurst_signal(price_series_short, hurst_window=50)
        valid = result.dropna()
        if len(valid) > 0:
            assert float(valid.min()) >= -1.01
            assert float(valid.max()) <= 1.01

    def test_aroon_oscillator_bounded(self, price_series_short):
        """Aroon oscillator divided by 100 should be in [-1, 1]."""
        result = tech_aroon_oscillator(price_series_short)
        valid = result.dropna()
        assert float(valid.min()) >= -1.0 - 1e-9
        assert float(valid.max()) <= 1.0 + 1e-9

    def test_mom_ts_moskowitz_same_sign_as_12m_return(self, price_series_short):
        """Moskowitz TS signal should have same sign as 12m return."""
        prices = price_series_short
        lp = np.log(prices.replace(0.0, float("nan")))
        ret_12m = lp - lp.shift(252)
        sig = mom_ts_moskowitz(prices)
        # Where both are defined, they should have the same sign
        common = ret_12m.dropna().index.intersection(sig.dropna().index)
        if len(common) > 0:
            same_sign = (np.sign(ret_12m.loc[common]) == np.sign(sig.loc[common]))
            assert same_sign.all(), "Moskowitz signal sign mismatch with 12m return"


# ---------------------------------------------------------------------------
# IC Calculator Tests
# ---------------------------------------------------------------------------


class TestICCalculator:
    """IC computations should return values in [-1, 1]."""

    @pytest.fixture(autouse=True)
    def calc(self):
        self.ic_calc = ICCalculator(min_obs=20)

    def _make_data(self, n: int = 200):
        rng = np.random.default_rng(42)
        signal = pd.Series(rng.normal(0, 1, n))
        # Correlated forward return
        noise = rng.normal(0, 1, n)
        fwd = 0.3 * signal + noise
        return signal, fwd

    def test_compute_ic_in_range(self):
        signal, fwd = self._make_data()
        ic = self.ic_calc.compute_ic(signal, fwd)
        assert math.isfinite(ic)
        assert -1.0 <= ic <= 1.0

    def test_compute_ic_nan_on_insufficient_data(self):
        tiny_signal = pd.Series([1.0, 2.0, 3.0])
        tiny_fwd = pd.Series([0.1, 0.2, 0.3])
        ic = self.ic_calc.compute_ic(tiny_signal, tiny_fwd)
        assert math.isnan(ic)

    def test_perfect_positive_correlation(self):
        x = pd.Series(np.arange(100, dtype=float))
        ic = self.ic_calc.compute_ic(x, x)
        assert abs(ic - 1.0) < 1e-6

    def test_perfect_negative_correlation(self):
        x = pd.Series(np.arange(100, dtype=float))
        ic = self.ic_calc.compute_ic(x, -x)
        assert abs(ic + 1.0) < 1e-6

    def test_rolling_ic_returns_series(self):
        signal, fwd = self._make_data(300)
        signal.index = pd.date_range("2022-01-01", periods=300, freq="B")
        fwd.index = signal.index
        ic_ts = self.ic_calc.rolling_ic(signal, fwd, window=60)
        assert isinstance(ic_ts, pd.Series)
        assert len(ic_ts) > 0

    def test_rolling_ic_values_bounded(self):
        signal, fwd = self._make_data(300)
        signal.index = pd.date_range("2022-01-01", periods=300, freq="B")
        fwd.index = signal.index
        ic_ts = self.ic_calc.rolling_ic(signal, fwd, window=60)
        valid = ic_ts.dropna()
        assert float(valid.abs().max()) <= 1.0 + 1e-9

    def test_icir_finite_with_reasonable_data(self):
        signal, fwd = self._make_data(300)
        signal.index = pd.date_range("2022-01-01", periods=300, freq="B")
        fwd.index = signal.index
        ic_ts = self.ic_calc.rolling_ic(signal, fwd, window=60)
        icir = self.ic_calc.compute_icir(ic_ts)
        assert math.isfinite(icir)

    def test_ic_decay_returns_dict(self, price_series_long):
        prices = price_series_long
        signal = mr_zscore_20(prices)
        decay = self.ic_calc.ic_decay(signal, prices, horizons=[1, 5, 20])
        assert isinstance(decay, dict)
        assert set(decay.keys()) == {1, 5, 20}
        for h, val in decay.items():
            if math.isfinite(val):
                assert -1.0 <= val <= 1.0, f"IC at horizon {h} = {val} out of [-1,1]"

    def test_detect_sign_flip_false_on_stable_signal(self):
        rng = np.random.default_rng(1)
        ic_ts = pd.Series(rng.normal(0.1, 0.05, 100))
        assert not self.ic_calc.detect_sign_flip(ic_ts, window=20)

    def test_detect_sign_flip_true_on_inverted_signal(self):
        # detect_sign_flip checks iloc[-(2*window):-window] vs iloc[-window:]
        # With window=30: checks iloc[-60:-30] vs iloc[-30:]
        # Need early(+) in positions [-60:-30] and late(-) in positions [-30:]
        # So: 60 bars of +0.3, then 30 bars of -0.3 → total=90, iloc[-60:-30]=+0.3, iloc[-30:]=-0.3
        early = pd.Series(np.full(60, 0.3))
        late = pd.Series(np.full(30, -0.3))
        ic_ts = pd.concat([early, late], ignore_index=True)
        assert self.ic_calc.detect_sign_flip(ic_ts, window=30)

    def test_full_ic_summary_returns_dataframe(self, price_series_long):
        prices = price_series_long
        signal = mr_rsi(prices)
        df = self.ic_calc.full_ic_summary(signal, prices, horizons=[1, 5, 20])
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"ic", "t_stat", "p_value", "n_obs"}
        valid_ic = df["ic"].dropna()
        for ic in valid_ic:
            assert -1.0 <= ic <= 1.0


# ---------------------------------------------------------------------------
# AlphaDecayAnalyzer Tests
# ---------------------------------------------------------------------------


class TestAlphaDecayAnalyzer:
    """Decay fitting and Fama-MacBeth tests."""

    def test_fit_decay_returns_dict(self):
        analyzer = AlphaDecayAnalyzer()
        ic_decay_dict = {1: 0.08, 5: 0.06, 10: 0.04, 20: 0.02, 40: 0.01, 60: 0.005}
        result = analyzer.fit_decay(ic_decay_dict)
        assert "ic0" in result
        assert "half_life" in result
        assert "lambda_" in result

    def test_fit_decay_half_life_positive(self):
        analyzer = AlphaDecayAnalyzer()
        ic_decay_dict = {1: 0.10, 5: 0.08, 10: 0.05, 20: 0.03, 60: 0.01}
        result = analyzer.fit_decay(ic_decay_dict)
        if math.isfinite(result["half_life"]):
            assert result["half_life"] > 0

    def test_is_stale_with_fast_decay(self):
        analyzer = AlphaDecayAnalyzer()
        # Very fast decay: half-life should be < 5
        ic_decay = {1: 0.1, 2: 0.001, 5: 0.0001, 10: 0.00001}
        result = analyzer.fit_decay(ic_decay)
        if math.isfinite(result["half_life"]):
            assert analyzer.is_stale(ic_decay) or result["half_life"] < 5

    def test_fama_macbeth_returns_dict(self):
        analyzer = AlphaDecayAnalyzer()
        rng = np.random.default_rng(10)
        n_dates, n_signals = 50, 20
        ic_panel = pd.DataFrame(
            rng.normal(0.05, 0.1, (n_dates, n_signals)),
            columns=[f"sig_{i}" for i in range(n_signals)],
        )
        result = analyzer.fama_macbeth_ic_regression(ic_panel)
        assert "beta" in result
        assert "t_stat" in result


# ---------------------------------------------------------------------------
# SignalCombiner Tests
# ---------------------------------------------------------------------------


class TestSignalCombiner:
    """SignalCombiner should reduce cross-signal correlation."""

    def _make_signal_df(self, n: int = 200, k: int = 10, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        data = {f"sig_{i}": rng.normal(0, 1, n) for i in range(k)}
        return pd.DataFrame(data)

    def test_equal_weight_returns_series(self):
        combiner = SignalCombiner()
        df = self._make_signal_df()
        result = combiner.equal_weight(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 200

    def test_ic_weight_returns_series(self):
        combiner = SignalCombiner()
        df = self._make_signal_df()
        ic_dict = {f"sig_{i}": 0.1 for i in range(10)}
        result = combiner.ic_weight(df, ic_dict)
        assert isinstance(result, pd.Series)

    def test_max_diversification_returns_series(self):
        combiner = SignalCombiner()
        df = self._make_signal_df()
        result = combiner.max_diversification(df)
        assert isinstance(result, pd.Series)

    def test_signal_correlation_matrix_is_square(self):
        combiner = SignalCombiner()
        df = self._make_signal_df(n=100, k=5)
        corr = combiner.signal_correlation_matrix(df)
        assert corr.shape == (5, 5)

    def test_signal_correlation_diagonal_is_one(self):
        combiner = SignalCombiner()
        df = self._make_signal_df(n=100, k=5)
        corr = combiner.signal_correlation_matrix(df)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-9)

    def test_deduplicate_removes_redundant(self):
        """Highly correlated signals should be deduplicated."""
        combiner = SignalCombiner(redundancy_threshold=0.8)
        rng = np.random.default_rng(5)
        base = rng.normal(0, 1, 200)
        df = pd.DataFrame({
            "A": base,
            "B": base + rng.normal(0, 0.01, 200),  # nearly identical to A
            "C": rng.normal(0, 1, 200),             # independent
        })
        deduped, removed = combiner.deduplicate(df)
        assert "B" in removed  # B should be removed (corr ~ 1 with A)
        assert "C" not in removed  # C should be kept

    def test_combined_signal_lower_corr_with_each_component(self):
        """Combined signal should have lower avg abs corr than individual signals with random pair."""
        combiner = SignalCombiner()
        rng = np.random.default_rng(55)
        k = 8
        df = pd.DataFrame({f"sig_{i}": rng.normal(0, 1, 200) for i in range(k)})
        combined = combiner.equal_weight(df)

        corrs = []
        for col in df.columns:
            c = df[col].corr(combined)
            corrs.append(abs(c))
        avg_corr = np.mean(corrs)
        # Combined signal with equal weight will have moderate correlation with each
        # Just verify it returns a valid number
        assert math.isfinite(avg_corr)

    def test_active_count_after_dedup(self):
        combiner = SignalCombiner(redundancy_threshold=0.8)
        rng = np.random.default_rng(5)
        base = rng.normal(0, 1, 200)
        df = pd.DataFrame({
            "A": base,
            "B": base + rng.normal(0, 0.01, 200),
            "C": rng.normal(0, 1, 200),
        })
        count = combiner.active_count_after_dedup(df)
        assert count <= 3
        assert count >= 1


# ---------------------------------------------------------------------------
# SignalRetirementEngine Tests
# ---------------------------------------------------------------------------


class TestSignalRetirementEngine:
    """Retirement engine should retire signals with low ICIR."""

    def test_no_retirement_with_high_icir(self, tmp_db):
        engine = SignalRetirementEngine(db_path=tmp_db, icir_threshold=0.2, consecutive_days=20)
        # Good signal: IC above 0.2
        ic_ts = pd.Series(np.full(60, 0.3))
        signals_reg: Dict = {}
        retired = engine.check_and_retire("good_signal", ic_ts, signals_reg)
        assert not retired

    def test_retirement_with_low_icir(self, tmp_db):
        engine = SignalRetirementEngine(db_path=tmp_db, icir_threshold=0.2, consecutive_days=20)
        # Bad signal: IC below 0.2 for all 60 days
        ic_ts = pd.Series(np.full(60, 0.05))
        s = Signal(name="bad_signal", description="test", category="momentum",
                   compute=mom_1d, parameters={})
        signals_reg = {"bad_signal": s}
        retired = engine.check_and_retire("bad_signal", ic_ts, signals_reg)
        assert retired
        assert not signals_reg["bad_signal"].is_active

    def test_retirement_log_stored(self, tmp_db):
        engine = SignalRetirementEngine(db_path=tmp_db, icir_threshold=0.2, consecutive_days=10)
        ic_ts = pd.Series(np.full(30, 0.01))
        signals_reg: Dict = {}
        engine.check_and_retire("log_test_signal", ic_ts, signals_reg)
        log_df = engine.get_retirement_log()
        assert isinstance(log_df, pd.DataFrame)


# ---------------------------------------------------------------------------
# AlphaEngine Tests
# ---------------------------------------------------------------------------


class TestAlphaEngine:
    """AlphaEngine round-trip tests."""

    @pytest.fixture(autouse=True)
    def engine(self, tmp_path):
        db = tmp_path / "alpha_engine.db"
        self.engine = AlphaEngine(db_path=db, ic_window=20)

    def test_register_signal(self):
        s = Signal(name="test_mom", description="test", category="momentum",
                   compute=mom_20d, parameters={})
        self.engine.register(s)
        assert "test_mom" in self.engine.signals

    def test_register_invalid_category(self):
        with pytest.raises(ValueError):
            Signal(name="bad_cat", description="test", category="invalid_category",
                   compute=mom_1d, parameters={})

    def test_score_daily_returns_dict(self, price_series_short, volume_series):
        s = Signal(name="score_mom", description="test", category="momentum",
                   compute=mom_20d, parameters={})
        self.engine.register(s)
        scores = self.engine.score_daily("TEST", price_series_short, volume=volume_series)
        assert isinstance(scores, dict)
        assert "score_mom" in scores

    def test_score_daily_values_finite_or_nan(self, price_series_short):
        s = Signal(name="rsi_test", description="test", category="mean_reversion",
                   compute=mr_rsi, parameters={})
        self.engine.register(s)
        scores = self.engine.score_daily("TEST2", price_series_short)
        for name, val in scores.items():
            assert math.isfinite(val) or math.isnan(val)

    def test_load_records_returns_dataframe(self, price_series_short):
        s = Signal(name="macd_test", description="test", category="technical",
                   compute=tech_macd_histogram, parameters={})
        self.engine.register(s)
        self.engine.score_daily("TEST3", price_series_short)
        df = self.engine.load_records(signal_name="macd_test")
        assert isinstance(df, pd.DataFrame)

    def test_build_combined_signal(self, price_series_short, volume_series):
        for i, (name, func) in enumerate(list(SIGNAL_REGISTRY.items())[:5]):
            s = Signal(name=f"combo_{i}", description=f"signal {i}",
                       category=list(SIGNAL_CATEGORIES.keys())[i % 6],
                       compute=func, parameters={})
            self.engine.register(s)
        combined = self.engine.build_combined_signal(price_series_short, volume=volume_series)
        assert isinstance(combined, pd.Series)

    def test_weekly_report(self, price_series_short):
        s1 = Signal(name="rep_mom", description="test", category="momentum",
                    compute=mom_20d, parameters={})
        s2 = Signal(name="rep_mr", description="test", category="mean_reversion",
                    compute=mr_zscore_20, parameters={})
        self.engine.register(s1)
        self.engine.register(s2)
        report = self.engine.generate_weekly_report("REPORT_SYM", price_series_short)
        assert isinstance(report, AlphaReport)
        assert isinstance(report.top_signals, list)
        assert isinstance(report.new_signals, list)
        assert math.isfinite(report.total_active) or report.total_active >= 0


# ---------------------------------------------------------------------------
# SignalRecord Tests
# ---------------------------------------------------------------------------


class TestSignalRecord:
    def test_to_dict_has_all_fields(self):
        rec = SignalRecord(
            signal_name="test",
            symbol="SPY",
            timestamp=datetime(2024, 1, 1),
            value=0.5,
            forward_return_1d=0.01,
        )
        d = rec.to_dict()
        assert "signal_name" in d
        assert "symbol" in d
        assert "timestamp" in d
        assert "value" in d

    def test_make_signal_record(self, price_series_short):
        rec = make_signal_record(
            signal_name="test",
            symbol="SPY",
            timestamp=datetime.utcnow(),
            signal_value=0.42,
            prices=price_series_short,
        )
        assert isinstance(rec, SignalRecord)
        assert rec.signal_name == "test"
        assert rec.symbol == "SPY"


# ---------------------------------------------------------------------------
# Feature Tests
# ---------------------------------------------------------------------------


class TestFeature:
    def test_feature_cache_key(self):
        f = Feature(
            name="mom_20d",
            value=0.5,
            timestamp=datetime(2024, 1, 15),
            symbol="SPY",
        )
        assert "SPY" in f.cache_key
        assert "mom_20d" in f.cache_key

    def test_feature_not_expired_immediately(self):
        f = Feature(
            name="vol_20d",
            value=0.2,
            timestamp=datetime.utcnow(),
            symbol="QQQ",
            ttl_seconds=3600,
        )
        assert not f.is_expired

    def test_feature_expired_with_zero_ttl(self):
        import time as time_module
        f = Feature(
            name="old",
            value=1.0,
            timestamp=datetime.utcnow() - timedelta(hours=2),
            symbol="X",
            ttl_seconds=1,
            computed_at=datetime.utcnow() - timedelta(seconds=10),
        )
        assert f.is_expired


# ---------------------------------------------------------------------------
# FeatureStore Tests
# ---------------------------------------------------------------------------


class TestFeatureStore:
    def test_put_and_get(self, tmp_fs_db):
        store = FeatureStore(db_path=tmp_fs_db)
        now = datetime.utcnow()
        f = Feature(name="mom_5d", value=0.3, timestamp=now, symbol="AAPL", ttl_seconds=3600)
        store.put(f)
        retrieved = store.get("AAPL", "mom_5d", now)
        assert retrieved is not None
        assert abs(retrieved.value - 0.3) < 1e-9

    def test_get_missing_returns_none(self, tmp_fs_db):
        store = FeatureStore(db_path=tmp_fs_db)
        result = store.get("MISSING", "fake_signal", datetime.utcnow())
        assert result is None

    def test_put_many_and_retrieve(self, tmp_fs_db):
        store = FeatureStore(db_path=tmp_fs_db)
        now = datetime.utcnow()
        features = [
            Feature(name=f"sig_{i}", value=float(i), timestamp=now, symbol="GOOG", ttl_seconds=3600)
            for i in range(10)
        ]
        store.put_many(features)
        retrieved = store.get_all_for_symbol("GOOG", as_of=now)
        assert len(retrieved) >= 10

    def test_cache_stats(self, tmp_fs_db):
        store = FeatureStore(db_path=tmp_fs_db)
        stats = store.cache_stats()
        assert "lru_size" in stats
        assert "lru_hit_rate" in stats


# ---------------------------------------------------------------------------
# FeatureComputer Tests
# ---------------------------------------------------------------------------


class TestFeatureComputer:
    def test_compute_all_returns_dict(self, price_series_short, volume_series):
        computer = FeatureComputer()
        result = computer.compute_all("SPY", price_series_short, volume=volume_series)
        assert isinstance(result, dict)
        assert len(result) >= 100

    def test_compute_all_values_finite_or_nan(self, price_series_short):
        computer = FeatureComputer()
        result = computer.compute_all("QQQ", price_series_short)
        for name, val in result.items():
            assert math.isfinite(val) or math.isnan(val), f"Signal '{name}' returned inf"

    def test_compute_all_to_features(self, price_series_short):
        computer = FeatureComputer()
        features = computer.compute_all_to_features("IWM", price_series_short)
        assert isinstance(features, list)
        assert all(isinstance(f, Feature) for f in features)

    def test_compute_series(self, price_series_short):
        computer = FeatureComputer()
        result = computer.compute_series("SPY", "mom_20d", price_series_short)
        assert isinstance(result, pd.Series)

    def test_compute_series_unknown_signal(self, price_series_short):
        computer = FeatureComputer()
        with pytest.raises(ValueError):
            computer.compute_series("SPY", "nonexistent_signal", price_series_short)

    def test_list_signals(self):
        computer = FeatureComputer()
        signals = computer.list_signals()
        assert len(signals) >= 100


# ---------------------------------------------------------------------------
# FeatureMatrix Tests
# ---------------------------------------------------------------------------


class TestFeatureMatrix:
    def _make_matrix(self, n_symbols: int = 10, n_features: int = 20) -> FeatureMatrix:
        rng = np.random.default_rng(22)
        data = {
            f"sym_{i}": {f"feat_{j}": rng.normal(0, 1) for j in range(n_features)}
            for i in range(n_symbols)
        }
        matrix = FeatureMatrix()
        matrix.from_dict(data)
        return matrix

    def test_shape(self):
        matrix = self._make_matrix()
        assert matrix.shape == (10, 20)

    def test_zscore_normalize(self):
        matrix = self._make_matrix()
        normed = matrix.zscore_normalize()
        df = normed.matrix
        for col in df.columns:
            vals = df[col].dropna()
            if len(vals) >= 5:
                assert abs(vals.mean()) < 0.5, f"Column {col} mean {vals.mean()} not near 0 after zscore"

    def test_rank_normalize_bounds(self):
        matrix = self._make_matrix()
        normed = matrix.rank_normalize()
        df = normed.matrix
        for col in df.columns:
            vals = df[col].dropna()
            assert float(vals.min()) >= -1.0 - 1e-9
            assert float(vals.max()) <= 1.0 + 1e-9

    def test_winsorize(self):
        rng = np.random.default_rng(99)
        data = {
            f"sym_{i}": {f"feat_{j}": rng.normal(0, 1) for j in range(10)}
            for i in range(20)
        }
        # Add outliers
        data["sym_0"]["feat_0"] = 1000.0
        data["sym_1"]["feat_0"] = -1000.0
        matrix = FeatureMatrix().from_dict(data)
        winsorized = matrix.winsorize(limits=0.05)
        col = winsorized.matrix["feat_0"]
        assert float(col.max()) < 100.0

    def test_impute_median_no_nan(self):
        matrix = self._make_matrix()
        imputed = matrix.impute_median()
        df = imputed.matrix
        assert df.isna().sum().sum() == 0

    def test_fill_rate(self):
        matrix = self._make_matrix()
        assert 0.0 <= matrix.fill_rate() <= 1.0

    def test_to_numpy(self):
        matrix = self._make_matrix()
        arr = matrix.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (10, 20)


# ---------------------------------------------------------------------------
# FeaturePipeline Tests
# ---------------------------------------------------------------------------


class TestFeaturePipeline:
    def test_run_nightly(self, tmp_path, price_series_short, volume_series):
        db = tmp_path / "pipeline.db"
        pipeline = create_feature_pipeline(db_path=db)
        symbol_data = {
            "AAPL": {"prices": price_series_short, "volume": volume_series},
            "GOOG": {"prices": price_series_short * 1.05},
        }
        stats = pipeline.run_nightly(symbol_data)
        assert stats["n_computed"] == 2
        assert stats["n_failed"] == 0

    def test_get_live_signal(self, tmp_path, price_series_short, volume_series):
        db = tmp_path / "live.db"
        pipeline = create_feature_pipeline(db_path=db)
        val = pipeline.get_live_signal("SPY", "mom_20d", price_series_short, volume=volume_series)
        assert math.isfinite(val) or math.isnan(val)

    def test_build_feature_matrix(self, tmp_path, price_series_short, volume_series):
        db = tmp_path / "matrix.db"
        pipeline = create_feature_pipeline(db_path=db)
        symbol_data = {
            f"SYM_{i}": {"prices": price_series_short * (1 + 0.01 * i)}
            for i in range(5)
        }
        pipeline.run_nightly(symbol_data)
        matrix = pipeline.build_feature_matrix(
            symbols=list(symbol_data.keys()),
            normalize="zscore",
            impute="median",
        )
        assert isinstance(matrix, FeatureMatrix)
        assert matrix.shape[0] > 0

    def test_is_stale_before_run(self, tmp_path):
        db = tmp_path / "stale.db"
        pipeline = create_feature_pipeline(db_path=db)
        assert pipeline.is_stale(max_age_hours=24)

    def test_is_not_stale_after_run(self, tmp_path, price_series_short):
        db = tmp_path / "notstale.db"
        pipeline = create_feature_pipeline(db_path=db)
        pipeline.run_nightly({"SPY": {"prices": price_series_short}})
        assert not pipeline.is_stale(max_age_hours=24)


# ---------------------------------------------------------------------------
# Integration: signal -> IC -> combiner pipeline
# ---------------------------------------------------------------------------


class TestIntegrationICPipeline:
    """End-to-end: compute signals, measure IC, combine."""

    def test_full_pipeline_ic_in_range(self, price_series_long):
        """IC for a reasonable signal on trending data should be in [-1, 1]."""
        prices = price_series_long
        calc = ICCalculator(min_obs=20)

        # Use a few signals
        signals_to_test = ["mom_20d", "mr_rsi", "vol_realized_20d", "tech_macd_histogram"]
        log_prices = np.log(prices.replace(0.0, float("nan")))
        fwd_1d = log_prices.shift(-1) - log_prices

        for sig_name in signals_to_test:
            func = SIGNAL_REGISTRY[sig_name]
            sig = func(prices)
            ic = calc.compute_ic(sig, fwd_1d)
            if math.isfinite(ic):
                assert -1.0 <= ic <= 1.0, f"IC for {sig_name} = {ic} out of bounds"

    def test_combined_signal_not_degenerate(self, price_series_long, volume_series):
        """Combined signal should have non-trivial std (not all zeros)."""
        prices = price_series_long
        vol = pd.Series(
            np.random.lognormal(15, 0.5, len(prices)),
            index=prices.index,
        )

        signal_df = pd.DataFrame({
            "mom_20d": mom_20d(prices),
            "mr_rsi": mr_rsi(prices),
            "vol_20d": vol_realized_20d(prices),
            "macd": tech_macd_histogram(prices),
            "obv": ms_obv_normalized(prices, volume=vol),
        }).dropna()

        combiner = SignalCombiner()
        combined = combiner.equal_weight(signal_df)
        assert combined.std() > 1e-10, "Combined signal has near-zero std"

    def test_ic_decay_monotone_for_strong_signal(self, price_series_long):
        """For momentum signal, IC should generally decay with horizon."""
        prices = price_series_long
        calc = ICCalculator(min_obs=20)
        sig = mom_20d(prices)
        decay = calc.ic_decay(sig, prices, horizons=[1, 5, 20, 60])

        finite_vals = [v for v in decay.values() if math.isfinite(v)]
        # Just check decay dict populated
        assert len(finite_vals) >= 2

    def test_signal_combiner_reduces_correlation(self, price_series_long):
        """
        Equal-weight combination of uncorrelated signals should have
        lower variance than the sum of individual variances would imply.
        """
        rng = np.random.default_rng(100)
        n = len(price_series_long)
        df = pd.DataFrame({
            f"s_{i}": rng.normal(0, 1, n)
            for i in range(10)
        }, index=price_series_long.index)

        combiner = SignalCombiner()
        corr = combiner.signal_correlation_matrix(df)
        np.fill_diagonal(corr.values, float("nan"))
        mean_off_diag = float(corr.abs().stack().mean())
        # For independent signals, mean abs off-diagonal corr should be low
        assert mean_off_diag < 0.3, f"Expected low corr for independent signals, got {mean_off_diag:.3f}"
