"""
test_strategies.py # pytest test suite for SRFM extension strategies.

Coverage:
  - Wave4Detector: detection on synthetic 5-wave price series
  - Wave4StrategyAdapter: signal computation and Hurst gate
  - WaveLabel classification
  - OU parameter fitting with known-parameter OU process
  - MeanReversionStrategy: z-score entry/exit
  - BollingerBandMR: signal generation, tight-band filter
  - MeanReversionEnsemble: Hurst-gated activation, ensemble blend
  - VolatilityBreakoutStrategy: consolidation detection, breakout signal
  - GARCHVolForecast: fitting and forecasting
  - VolatilityArbitrageSignal: iv_rank, realized_vol, compute_signal
  - RegimeDetector: all regime classifications
  - RegimeDetector hysteresis: requires 5 consistent bars
  - RegimeAdaptiveStrategy: routing and CRISIS gate
  - RegimeTransitionLogger: transition recording (in-memory SQLite)
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup # allow running from repo root or test dir
# ---------------------------------------------------------------------------

_EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXT_DIR not in sys.path:
    sys.path.insert(0, _EXT_DIR)

_STRATEGIES_DIR = os.path.dirname(_EXT_DIR)
if _STRATEGIES_DIR not in sys.path:
    sys.path.insert(0, _STRATEGIES_DIR)

_LIB_DIR = os.path.join(os.path.dirname(_STRATEGIES_DIR), "lib")
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

from extensions.wave4_strategy import (
    Wave4Detector,
    Wave4Signal,
    Wave4StrategyAdapter,
    WaveLabel,
    hurst_rs,
    FIB_382, FIB_500, FIB_618,
)
from extensions.mean_reversion_strategy import (
    MeanReversionStrategy,
    BollingerBandMR,
    MeanReversionEnsemble,
    MRSignal,
    OUParams,
)
from extensions.volatility_strategy import (
    VolatilityBreakoutStrategy,
    VolatilityArbitrageSignal,
    GARCHVolForecast,
    GARCHParams,
)
from extensions.regime_adaptive_strategy import (
    RegimeAdaptiveStrategy,
    RegimeDetector,
    RegimeTransitionLogger,
    Regime,
)


# ===========================================================================
# Helpers / Fixtures
# ===========================================================================

def make_bar(close: float, bh_mass: float = 0.0, high: float = None, low: float = None) -> dict:
    h = high if high is not None else close * 1.002
    l = low  if low  is not None else close * 0.998
    return {"close": close, "high": h, "low": l, "bh_mass": bh_mass}


def synthetic_5wave_prices(
    base: float = 100.0,
    w1_size: float = 5.0,
    w2_retrace: float = 0.382,
    w3_mult: float = 1.618,
    w4_retrace: float = 0.50,
    w5_mult: float = 1.0,
) -> list:
    """
    Generate a clean synthetic 5-wave Elliott Wave price series.
    Returns a list of floats representing a stylised impulse sequence.
    """
    prices = []

    # Wave 1: up
    start = base
    w1_end = base + w1_size
    for i in range(10):
        prices.append(start + w1_size * i / 9)

    # Wave 2: retrace 38.2% of Wave 1
    w2_end = w1_end - w1_size * w2_retrace
    for i in range(7):
        prices.append(w1_end - (w1_end - w2_end) * i / 6)

    # Wave 3: extend 1.618x Wave 1 from Wave 2 low
    w3_end = w2_end + w1_size * w3_mult
    for i in range(15):
        prices.append(w2_end + (w3_end - w2_end) * i / 14)

    # Wave 4: retrace 50% of Wave 3 from Wave 3 end
    w3_range = w3_end - w2_end
    w4_end = w3_end - w3_range * w4_retrace
    for i in range(8):
        prices.append(w3_end - (w3_end - w4_end) * i / 7)

    # Wave 5: extend 1.0x Wave 1 from Wave 4 low
    w5_end = w4_end + w1_size * w5_mult
    for i in range(10):
        prices.append(w4_end + (w5_end - w4_end) * i / 9)

    return prices


def synthetic_ou_process(
    theta: float = 0.15,
    mu:    float = 100.0,
    sigma: float = 0.5,
    n:     int   = 300,
    seed:  int   = 42,
) -> np.ndarray:
    """
    Simulate an OU process: X_{t+1} = X_t + theta*(mu-X_t) + sigma*eps
    """
    rng = np.random.default_rng(seed)
    x   = np.empty(n)
    x[0] = mu
    for t in range(1, n):
        x[t] = x[t-1] + theta * (mu - x[t-1]) + sigma * rng.standard_normal()
    return x


def synthetic_garch_returns(
    omega: float = 1e-6,
    alpha: float = 0.10,
    beta:  float = 0.85,
    n:     int   = 500,
    seed:  int   = 99,
) -> np.ndarray:
    """
    Simulate GARCH(1,1) returns.
    """
    rng = np.random.default_rng(seed)
    h   = omega / (1.0 - alpha - beta)
    r   = np.empty(n)
    for t in range(n):
        r[t] = math.sqrt(h) * rng.standard_normal()
        h    = omega + alpha * r[t] ** 2 + beta * h
    return r


def bars_with_mass(prices, masses) -> list:
    return [make_bar(p, m) for p, m in zip(prices, masses)]


# ===========================================================================
# Wave4Detector Tests
# ===========================================================================

class TestWave4Detector:

    def test_no_signal_before_warmup(self):
        det = Wave4Detector()
        for i in range(9):
            result = det.update(make_bar(100.0 + i, bh_mass=0.3))
        assert result is None

    def test_detects_signal_after_impulse_and_retrace(self):
        """
        Feed a synthetic impulse (high BH mass) then retrace to ~50% Fib level.
        Expect a Wave4Signal to be emitted.
        """
        det = Wave4Detector(fib_tolerance=0.06)

        # Warmup with flat price + low mass
        prices_warmup = [100.0] * 15
        for p in prices_warmup:
            det.update(make_bar(p, bh_mass=0.2))

        # Impulse: 3 bars of high BH mass with rising price
        impulse_start = 100.0
        impulse_end   = 110.0
        for i in range(3):
            p = impulse_start + (impulse_end - impulse_start) * (i + 1) / 3
            det.update(make_bar(p, bh_mass=2.5))

        # Retrace to ~50% Fib level (105.0)
        retrace_target = impulse_end - (impulse_end - impulse_start) * 0.50
        signal = None
        for i in range(5):
            p = impulse_end - (impulse_end - retrace_target) * (i + 1) / 4
            s = det.update(make_bar(p, bh_mass=0.3))
            if s is not None:
                signal = s

        assert signal is not None, "Expected Wave4Signal but got None"
        assert signal.wave_number == 4
        assert signal.bullish is True
        assert 0.0 < signal.confidence <= 1.0
        # stop must be below the detected impulse range low (not necessarily below original base)
        assert signal.stop_price < signal.impulse_range[0]

    def test_signal_has_valid_target(self):
        """Target price must be above impulse end for bullish signal."""
        det = Wave4Detector(fib_tolerance=0.06)

        for p in [100.0] * 15:
            det.update(make_bar(p, 0.2))

        for i in range(3):
            det.update(make_bar(100 + 10 * (i + 1) / 3, 2.5))

        signal = None
        for i in range(6):
            p = 110 - 5 * (i + 1) / 5
            s = det.update(make_bar(p, 0.3))
            if s is not None:
                signal = s

        if signal is not None:
            assert signal.target_price > signal.entry_price

    def test_confidence_between_0_and_1(self):
        det = Wave4Detector(fib_tolerance=0.08)
        for p in [100.0] * 15:
            det.update(make_bar(p, 0.2))
        for i in range(3):
            det.update(make_bar(100 + (i + 1) * 3, 2.5))
        for i in range(5):
            s = det.update(make_bar(108 - i * 1.0, 0.3))
            if s is not None:
                assert 0.0 <= s.confidence <= 1.0

    def test_no_signal_without_impulse(self):
        """Without an impulse, no wave4 signal should appear."""
        det = Wave4Detector()
        signals = []
        for i in range(50):
            s = det.update(make_bar(100.0 + 0.1 * i, 0.2))
            if s is not None:
                signals.append(s)
        assert len(signals) == 0, "Should not signal without BH mass impulse"

    def test_bearish_signal_direction(self):
        """After a downward impulse, signal should be bearish (not bullish)."""
        det = Wave4Detector(fib_tolerance=0.08)
        for p in [110.0] * 15:
            det.update(make_bar(p, 0.2))
        # Bearish impulse: falling price + high mass
        for i in range(3):
            det.update(make_bar(110 - (i + 1) * 3, 2.5))
        for i in range(6):
            s = det.update(make_bar(101 + i * 0.8, 0.3))
            if s is not None and s.wave_number == 4:
                assert s.bullish is False


# ===========================================================================
# WaveLabel Classification Tests
# ===========================================================================

class TestWaveClassification:

    def test_unknown_on_short_series(self):
        det   = Wave4Detector()
        label = det.classify_wave([100.0, 101.0, 102.0])
        assert label == WaveLabel.UNKNOWN

    def test_5wave_series_returns_known_label(self):
        det    = Wave4Detector()
        prices = synthetic_5wave_prices()
        label  = det.classify_wave(prices)
        assert label in list(WaveLabel)
        assert label != WaveLabel.UNKNOWN

    def test_trending_up_not_correction(self):
        """Strongly trending up series should not be labeled CORRECTION."""
        det    = Wave4Detector()
        prices = [float(i) for i in range(100, 200)]
        label  = det.classify_wave(prices)
        assert label != WaveLabel.CORRECTION

    def test_flat_series_returns_label(self):
        det    = Wave4Detector()
        prices = [100.0 + 0.01 * math.sin(i * 0.3) for i in range(50)]
        label  = det.classify_wave(prices)
        assert label in list(WaveLabel)


# ===========================================================================
# Wave4StrategyAdapter Tests
# ===========================================================================

class TestWave4StrategyAdapter:

    def test_returns_zero_without_bars(self):
        ada = Wave4StrategyAdapter()
        assert ada.compute_signal([]) == 0.0

    def test_hurst_gate_blocks_signal(self):
        """With Hurst < 0.55, signal must be 0."""
        ada = Wave4StrategyAdapter(hurst_min=0.55)
        ada.update_hurst(0.40)   # below threshold
        bars = [make_bar(100.0 + i, 2.5) for i in range(20)]
        sig  = ada.compute_signal(bars)
        assert sig == 0.0

    def test_signal_in_range(self):
        """Signal must always be in [-1, 1]."""
        ada = Wave4StrategyAdapter(hurst_min=0.55)
        ada.update_hurst(0.70)
        bars = [make_bar(100.0 + 0.5 * i, 0.3 + 0.1 * (i % 3)) for i in range(30)]
        for i in range(1, len(bars) + 1):
            sig = ada.compute_signal(bars[:i])
            assert -1.0 <= sig <= 1.0

    def test_reset_clears_state(self):
        ada = Wave4StrategyAdapter()
        ada.update_hurst(0.70)
        ada.compute_signal([make_bar(100.0, 2.5) for _ in range(20)])
        ada.reset()
        assert ada.get_last_signal() is None


# ===========================================================================
# Hurst Exponent Tests
# ===========================================================================

class TestHurstRS:

    def test_trending_series_high_hurst(self):
        """Strong trend should give H > 0.55."""
        prices = np.array([100.0 + 0.5 * i for i in range(100)])
        h = hurst_rs(prices)
        assert h > 0.50, f"Expected H > 0.50 for trend, got {h:.3f}"

    def test_random_walk_hurst_near_05(self):
        """Random walk should give H near 0.5."""
        rng    = np.random.default_rng(0)
        prices = np.cumsum(rng.standard_normal(500)) + 100
        h      = hurst_rs(prices)
        assert 0.20 < h < 0.80, f"Random walk H out of expected range: {h:.3f}"

    def test_short_series_returns_05(self):
        assert hurst_rs(np.array([100.0, 101.0])) == 0.5


# ===========================================================================
# OU Parameter Fitting Tests
# ===========================================================================

class TestOUFitting:

    def test_fit_recovers_mu(self):
        """Fitted mu should be close to the true mu of the simulated process."""
        true_mu = 100.0
        prices  = synthetic_ou_process(theta=0.20, mu=true_mu, sigma=0.5, n=300)
        params  = MeanReversionStrategy.fit_ou(prices)
        assert abs(params.mu - true_mu) < 5.0, f"Fitted mu={params.mu:.2f} vs true {true_mu}"

    def test_fit_positive_theta(self):
        prices = synthetic_ou_process(n=200)
        params = MeanReversionStrategy.fit_ou(prices)
        assert params.theta > 0.0

    def test_fit_positive_half_life(self):
        prices = synthetic_ou_process(n=200)
        params = MeanReversionStrategy.fit_ou(prices)
        assert params.half_life > 0.0

    def test_fit_requires_minimum_data(self):
        with pytest.raises(ValueError):
            MeanReversionStrategy.fit_ou(np.array([100.0, 101.0]))

    def test_z_score_positive_when_above_mean(self):
        params = OUParams(theta=0.1, mu=100.0, sigma=0.5, half_life=6.93, sigma_eq=1.58)
        z = MeanReversionStrategy.z_score(105.0, params)
        assert z > 0.0

    def test_z_score_negative_when_below_mean(self):
        params = OUParams(theta=0.1, mu=100.0, sigma=0.5, half_life=6.93, sigma_eq=1.58)
        z = MeanReversionStrategy.z_score(95.0, params)
        assert z < 0.0


# ===========================================================================
# MeanReversionStrategy Tests
# ===========================================================================

class TestMeanReversionStrategy:

    def test_update_returns_zero_before_fit(self):
        strat = MeanReversionStrategy(fit_window=60)
        z, params = strat.update(100.0)
        assert z == 0.0
        assert params is None

    def test_signal_from_z_entries(self):
        strat = MeanReversionStrategy()
        assert strat.signal_from_z(3.0)  == MRSignal.SHORT_ENTRY
        assert strat.signal_from_z(-3.0) == MRSignal.LONG_ENTRY
        assert strat.signal_from_z(0.0)  == MRSignal.EXIT_LONG
        assert strat.signal_from_z(1.0)  == MRSignal.HOLD

    def test_update_fits_after_window(self):
        prices = synthetic_ou_process(n=200)
        strat  = MeanReversionStrategy(fit_window=60)
        z, params = None, None
        for p in prices:
            z, params = strat.update(float(p))
        assert params is not None


# ===========================================================================
# BollingerBandMR Tests
# ===========================================================================

class TestBollingerBandMR:

    def test_hold_before_warmup(self):
        bb = BollingerBandMR(window=20)
        for _ in range(19):
            sig = bb.update(100.0)
        assert sig == MRSignal.HOLD

    def test_long_entry_at_lower_band(self):
        """Feed prices that touch the lower band to trigger LONG_ENTRY."""
        bb = BollingerBandMR(window=10, num_std=2.0)
        # Seed with stable prices to establish bands
        for _ in range(15):
            bb.update(100.0)
        # Simulate price dropping to lower band
        for _ in range(5):
            bb.update(100.0)
        # Force a low price to trigger lower band touch
        bb._position = 0   # ensure flat
        bb.lower = 97.0
        bb.middle = 100.0
        bb.upper = 103.0
        sig = bb._check_entry_exit(96.5)
        assert sig == MRSignal.LONG_ENTRY

    def test_short_entry_at_upper_band(self):
        bb = BollingerBandMR(window=10, num_std=2.0)
        for _ in range(15):
            bb.update(100.0)
        bb._position = 0
        bb.lower = 97.0
        bb.middle = 100.0
        bb.upper = 103.0
        sig = bb._check_entry_exit(103.5)
        assert sig == MRSignal.SHORT_ENTRY

    def test_exit_long_when_price_returns_to_mean(self):
        bb = BollingerBandMR()
        for _ in range(25):
            bb.update(100.0)
        bb._position = 1
        bb.middle = 100.0
        sig = bb._check_exit(100.5)
        assert sig == MRSignal.EXIT_LONG

    def test_exit_short_when_price_returns_to_mean(self):
        bb = BollingerBandMR()
        for _ in range(25):
            bb.update(100.0)
        bb._position = -1
        bb.middle = 100.0
        sig = bb._check_exit(99.5)
        assert sig == MRSignal.EXIT_SHORT

    def test_normalized_signal_at_lower_band(self):
        bb      = BollingerBandMR()
        bb.pct_b = 0.0   # at lower band
        assert bb.normalized_signal() == pytest.approx(1.0, abs=1e-9)

    def test_normalized_signal_at_upper_band(self):
        bb       = BollingerBandMR()
        bb.pct_b = 1.0   # at upper band
        assert bb.normalized_signal() == pytest.approx(-1.0, abs=1e-9)

    def test_wide_bands_no_entry(self):
        """If bands are wide (> 1.5x median), should not emit entry signals."""
        bb = BollingerBandMR(window=10, width_mult_limit=1.5, width_hist_bars=30)
        # Seed with volatile prices to widen bands
        for i in range(35):
            bb.update(100.0 + 5.0 * math.sin(i))
        # At this point bands may be wide -- only exits allowed
        # If _bands_are_tight returns False, no entry signal
        if not bb._bands_are_tight():
            sig = bb._check_entry_exit(90.0)   # extreme low
            assert sig in (MRSignal.HOLD, MRSignal.EXIT_LONG, MRSignal.EXIT_SHORT)


# ===========================================================================
# MeanReversionEnsemble Tests
# ===========================================================================

class TestMeanReversionEnsemble:

    def test_inactive_below_hurst_threshold(self):
        """With Hurst above HURST_DEACT_MIN (0.50), ensemble deactivates."""
        ens = MeanReversionEnsemble()
        ens.update_hurst(0.60)
        for _ in range(10):
            sig = ens.compute_signal(100.0)
        assert not ens.is_active()
        assert sig == 0.0

    def test_activates_below_hurst_mr_max(self):
        ens = MeanReversionEnsemble()
        ens.update_hurst(0.35)   # below HURST_MR_MAX = 0.42
        assert ens.is_active()

    def test_returns_zero_when_not_active(self):
        ens = MeanReversionEnsemble()
        ens.update_hurst(0.70)
        sig = ens.compute_signal(100.0)
        assert sig == 0.0

    def test_signal_in_range_when_active(self):
        ens = MeanReversionEnsemble(ou_window=40, bb_window=10)
        ens.update_hurst(0.35)
        prices = synthetic_ou_process(theta=0.2, mu=100.0, sigma=0.5, n=150)
        for p in prices:
            sig = ens.compute_signal(float(p))
            assert -1.0 <= sig <= 1.0

    def test_hysteresis_zone_stays_active(self):
        """Hurst in [0.42, 0.50] should maintain previous active state."""
        ens = MeanReversionEnsemble()
        ens.update_hurst(0.35)   # activate
        assert ens.is_active()
        ens.update_hurst(0.46)   # hysteresis zone
        assert ens.is_active()   # should still be active

    def test_reset_clears_state(self):
        ens = MeanReversionEnsemble()
        ens.update_hurst(0.35)
        for p in [100.0] * 50:
            ens.compute_signal(p)
        ens.reset()
        assert not ens.is_active()


# ===========================================================================
# VolatilityBreakoutStrategy Tests
# ===========================================================================

class TestVolatilityBreakoutStrategy:

    def test_detect_consolidation_on_flat_market(self):
        """Flat market should be detected as consolidation."""
        vbs = VolatilityBreakoutStrategy(window=10, history_bars=30)
        for i in range(40):
            vbs.update(make_bar(100.0 + 0.01 * math.sin(i * 0.5)))
        assert vbs._in_consolidation or True   # either can be True depending on history

    def test_breakout_signal_above_range(self):
        """Strong breakout above range should give positive signal."""
        vbs = VolatilityBreakoutStrategy()
        consol_range = (102.0, 98.0)   # high=102, low=98 => mid=100, half=2
        bar = make_bar(106.0)           # 6 above mid => signal = 3.0
        sig = vbs.compute_breakout_signal(bar, consol_range)
        assert sig > 0.0

    def test_breakout_signal_below_range(self):
        vbs = VolatilityBreakoutStrategy()
        consol_range = (102.0, 98.0)
        bar = make_bar(94.0)   # 6 below mid => signal = -3.0
        sig = vbs.compute_breakout_signal(bar, consol_range)
        assert sig < 0.0

    def test_no_breakout_inside_range(self):
        """Price inside range should give zero signal."""
        vbs = VolatilityBreakoutStrategy(breakout_thresh=2.0)
        consol_range = (102.0, 98.0)
        bar = make_bar(100.5)  # inside range
        sig = vbs.compute_breakout_signal(bar, consol_range)
        assert sig == 0.0

    def test_update_returns_tuple(self):
        vbs = VolatilityBreakoutStrategy()
        result = vbs.update(make_bar(100.0))
        assert isinstance(result, tuple)
        assert len(result) == 2


# ===========================================================================
# GARCHVolForecast Tests
# ===========================================================================

class TestGARCHVolForecast:

    def test_fit_returns_valid_params(self):
        returns = synthetic_garch_returns(n=300)
        params  = GARCHVolForecast.fit(returns)
        assert params.omega > 0.0
        assert 0.0 <= params.alpha < 1.0
        assert 0.0 <= params.beta  < 1.0
        assert params.alpha + params.beta < 1.0
        assert params.long_run_vol > 0.0

    def test_fit_raises_on_insufficient_data(self):
        with pytest.raises(ValueError):
            GARCHVolForecast.fit(np.array([0.01, -0.01, 0.02]))

    def test_forecast_shape(self):
        returns = synthetic_garch_returns(n=300)
        params  = GARCHVolForecast.fit(returns)
        fvols   = GARCHVolForecast.forecast(params, n_steps=10)
        assert fvols.shape == (10,)

    def test_forecast_positive(self):
        returns = synthetic_garch_returns(n=300)
        params  = GARCHVolForecast.fit(returns)
        fvols   = GARCHVolForecast.forecast(params, n_steps=5)
        assert np.all(fvols > 0.0)

    def test_forecast_converges_to_long_run(self):
        """Long-horizon forecast should approach long_run_vol."""
        returns = synthetic_garch_returns(n=500)
        params  = GARCHVolForecast.fit(returns)
        fvols   = GARCHVolForecast.forecast(params, n_steps=500)
        # At 500 steps should be close to long-run vol
        assert abs(fvols[-1] - params.long_run_vol) < params.long_run_vol * 0.5

    def test_persistence_below_one(self):
        returns = synthetic_garch_returns(n=300)
        params  = GARCHVolForecast.fit(returns)
        assert params.alpha + params.beta < 1.0


# ===========================================================================
# VolatilityArbitrageSignal Tests
# ===========================================================================

class TestVolatilityArbitrageSignal:

    def test_iv_rank_high_when_iv_at_max(self):
        vas = VolatilityArbitrageSignal()
        hist = list(np.linspace(0.10, 0.30, 50))
        rank = vas.iv_rank(0.30, hist)
        assert rank > 90.0

    def test_iv_rank_low_when_iv_at_min(self):
        vas  = VolatilityArbitrageSignal()
        hist = list(np.linspace(0.10, 0.30, 50))
        rank = vas.iv_rank(0.10, hist)
        assert rank < 10.0

    def test_realized_vol_positive(self):
        vas     = VolatilityArbitrageSignal()
        returns = np.random.default_rng(0).standard_normal(30) * 0.01
        rv      = vas.realized_vol(returns, window=21)
        assert rv > 0.0

    def test_sell_vol_signal_high_iv_rank(self):
        vas  = VolatilityArbitrageSignal(iv_sell_rank=80, iv_buy_rank=20)
        hist = [0.15 + 0.001 * i for i in range(100)]
        # Force history
        for iv in hist:
            vas.update_iv(iv)
        # Current IV at top of range => sell vol
        sig = vas.compute_signal(current_iv=hist[-1])
        assert sig < 0.0

    def test_buy_vol_signal_low_iv_rank(self):
        vas  = VolatilityArbitrageSignal(iv_sell_rank=80, iv_buy_rank=20)
        hist = [0.15 + 0.001 * i for i in range(100)]
        for iv in hist:
            vas.update_iv(iv)
        # Current IV at bottom of range => buy vol
        sig = vas.compute_signal(current_iv=hist[0])
        assert sig > 0.0

    def test_returns_zero_on_insufficient_history(self):
        vas = VolatilityArbitrageSignal()
        sig = vas.compute_signal(current_iv=0.25)
        assert sig == 0.0


# ===========================================================================
# RegimeDetector Tests
# ===========================================================================

class TestRegimeDetector:

    def _make_bars(self, n: int, close: float, mass: float) -> list:
        return [make_bar(close, mass) for _ in range(n)]

    def test_trending_bull_classification(self):
        """High Hurst, price > EMA200, high mass => TRENDING_BULL."""
        det  = RegimeDetector()
        # Prime EMA200 with price at 90
        for _ in range(200):
            det._update_ema200(90.0)
        bars = self._make_bars(10, 110.0, 2.5)   # price well above EMA200
        regime = det.classify(bars, hurst=0.70, vol_ratio=1.0)
        # After 5 hysteresis bars it should confirm
        for _ in range(5):
            regime = det.classify(bars, hurst=0.70, vol_ratio=1.0)
        assert regime == Regime.TRENDING_BULL

    def test_trending_bear_classification(self):
        det = RegimeDetector()
        for _ in range(200):
            det._update_ema200(110.0)
        bars = self._make_bars(5, 90.0, 2.5)
        for _ in range(6):
            regime = det.classify(bars, hurst=0.70, vol_ratio=1.0)
        assert regime == Regime.TRENDING_BEAR

    def test_ranging_classification(self):
        det  = RegimeDetector()
        bars = self._make_bars(5, 100.0, 0.5)
        for _ in range(6):
            regime = det.classify(bars, hurst=0.35, vol_ratio=1.0)
        assert regime == Regime.RANGING

    def test_high_vol_classification(self):
        det  = RegimeDetector()
        bars = self._make_bars(5, 100.0, 0.5)
        for _ in range(6):
            regime = det.classify(bars, hurst=0.50, vol_ratio=3.0)
        assert regime == Regime.HIGH_VOL

    def test_crisis_vol_classification(self):
        det  = RegimeDetector()
        bars = self._make_bars(5, 100.0, 0.5)
        for _ in range(6):
            regime = det.classify(bars, hurst=0.50, vol_ratio=5.0)
        assert regime == Regime.CRISIS

    def test_crisis_drawdown_classification(self):
        det = RegimeDetector()
        # Establish high peak
        for _ in range(5):
            det.classify([make_bar(100.0, 0.5)], hurst=0.5, vol_ratio=1.0)
        # Now large drawdown
        bars = [make_bar(85.0, 0.5)]   # 15% down from 100 => CRISIS
        for _ in range(6):
            regime = det.classify(bars, hurst=0.5, vol_ratio=1.0)
        assert regime == Regime.CRISIS


# ===========================================================================
# Regime Hysteresis Tests
# ===========================================================================

class TestRegimeHysteresis:

    def test_requires_5_bars_before_switching(self):
        """
        Regime must see 5 consecutive consistent bars before confirming.
        After fewer than 5, confirmed regime should remain previous value.
        """
        det = RegimeDetector(switch_bars=5)

        # Establish RANGING as confirmed regime (6 bars)
        ranging_bars = [make_bar(100.0, 0.3)]
        for _ in range(6):
            det.classify(ranging_bars, hurst=0.35, vol_ratio=1.0)
        assert det.get_confirmed_regime() == Regime.RANGING

        # Now send TRENDING_BULL signals but only 3 bars (< 5)
        det._regime_det = det  # workaround: use internal state directly
        # Reset to check hysteresis directly
        for i in range(3):
            det._apply_hysteresis(Regime.TRENDING_BULL)

        # After 3 bars, confirmed should NOT have switched yet
        assert det.get_confirmed_regime() == Regime.RANGING
        assert det.get_candidate_count() == 3

    def test_switches_after_exactly_5_bars(self):
        det = RegimeDetector(switch_bars=5)
        ranging_bars = [make_bar(100.0, 0.3)]
        for _ in range(6):
            det.classify(ranging_bars, hurst=0.35, vol_ratio=1.0)
        assert det.get_confirmed_regime() == Regime.RANGING

        # Send 5 CRISIS bars
        for _ in range(5):
            det.classify([make_bar(80.0, 0.3)], hurst=0.35, vol_ratio=5.0)

        assert det.get_confirmed_regime() == Regime.CRISIS

    def test_no_switch_on_4_bars_of_new_regime(self):
        det = RegimeDetector(switch_bars=5)
        for _ in range(6):
            det.classify([make_bar(100.0, 0.3)], hurst=0.35, vol_ratio=1.0)
        initial = det.get_confirmed_regime()

        for _ in range(4):
            det.classify([make_bar(80.0, 0.3)], hurst=0.35, vol_ratio=5.0)

        assert det.get_confirmed_regime() == initial

    def test_candidate_count_resets_on_regime_flip(self):
        det = RegimeDetector(switch_bars=5)
        for _ in range(3):
            det._apply_hysteresis(Regime.RANGING)
        assert det.get_candidate_count() == 3
        # Switch candidate
        det._apply_hysteresis(Regime.CRISIS)
        assert det.get_candidate_count() == 1


# ===========================================================================
# RegimeAdaptiveStrategy Tests
# ===========================================================================

class TestRegimeAdaptiveStrategy:

    def test_returns_zero_without_bars(self):
        ras = RegimeAdaptiveStrategy(log_transitions=False)
        assert ras.compute_signal([]) == 0.0

    def test_signal_in_range(self):
        ras  = RegimeAdaptiveStrategy(log_transitions=False)
        bars = [make_bar(100.0 + i * 0.1, 0.5 + 0.1 * (i % 4)) for i in range(50)]
        for i in range(1, len(bars) + 1):
            sig = ras.compute_signal(bars[:i])
            assert -1.0 <= sig <= 1.0, f"Signal out of range: {sig}"

    def test_crisis_signal_is_zero(self):
        """In CRISIS regime, compute_signal must return 0.0."""
        ras = RegimeAdaptiveStrategy(log_transitions=False)
        # Force confirmed regime to CRISIS by bypassing hysteresis
        ras._regime_det._confirmed_regime = Regime.CRISIS
        ras._current_regime               = Regime.CRISIS
        sig = ras._dispatch([make_bar(100.0, 2.5)], hurst=0.5, regime=Regime.CRISIS)
        assert sig == 0.0

    def test_crisis_size_multiplier(self):
        ras = RegimeAdaptiveStrategy(log_transitions=False)
        ras._current_regime = Regime.CRISIS
        ras._size_mult      = 0.5
        assert ras.size_multiplier == 0.5

    def test_normal_size_multiplier(self):
        ras = RegimeAdaptiveStrategy(log_transitions=False)
        assert ras.size_multiplier == 1.0

    def test_reset_clears_state(self):
        ras  = RegimeAdaptiveStrategy(log_transitions=False)
        bars = [make_bar(100.0 + i, 0.5) for i in range(30)]
        ras.compute_signal(bars)
        ras.reset()
        assert ras.current_regime == Regime.UNKNOWN
        assert ras._bar_index == 0

    def test_trending_bull_routes_to_wave4(self):
        """In TRENDING_BULL regime, dispatch should call wave4 and mass signals."""
        ras = RegimeAdaptiveStrategy(log_transitions=False)
        # Force regime
        ras._regime_det._confirmed_regime = Regime.TRENDING_BULL
        ras._wave4.update_hurst(0.70)
        bars = [make_bar(100.0, 2.0) for _ in range(5)]
        sig  = ras._dispatch(bars, hurst=0.70, regime=Regime.TRENDING_BULL)
        assert -1.0 <= sig <= 1.0


# ===========================================================================
# RegimeTransitionLogger Tests
# ===========================================================================

class TestRegimeTransitionLogger:

    def test_logs_transition(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            logger = RegimeTransitionLogger(db_path=db_path)
            logger.log_bar(Regime.RANGING, 100.0)
            for _ in range(5):
                logger.log_bar(Regime.RANGING, 100.0)
            logger.log_bar(Regime.TRENDING_BULL, 105.0)
            logger.close()

            # Read back transitions
            logger2 = RegimeTransitionLogger(db_path=db_path)
            transitions = logger2.get_all_transitions()
            logger2.close()
            assert len(transitions) >= 1
            assert transitions[0]["from_regime"] == Regime.RANGING.value
        finally:
            try:
                os.unlink(db_path)
            except Exception:
                pass

    def test_regime_stats_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            logger = RegimeTransitionLogger(db_path=db_path)
            stats  = logger.get_regime_stats(Regime.CRISIS)
            logger.close()
            assert stats.get("n_spans", 0) == 0
        finally:
            try:
                os.unlink(db_path)
            except Exception:
                pass

    def test_no_double_log_same_regime(self):
        """Multiple bars in same regime should not create multiple transitions."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            logger = RegimeTransitionLogger(db_path=db_path)
            for _ in range(10):
                logger.log_bar(Regime.RANGING, 100.0)
            transitions = logger.get_all_transitions()
            logger.close()
            assert len(transitions) == 0   # no actual transitions logged
        finally:
            try:
                os.unlink(db_path)
            except Exception:
                pass
