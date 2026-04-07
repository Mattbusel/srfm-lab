"""
test_alt_data.py -- Production test suite for alt_data research modules.

Covers:
  - macro_regime: regime classification, yield curve, credit spreads
  - on_chain_advanced: realized price bands, exchange flows, miner signal,
    stablecoin ratio, NVT signal
  - satellite_web: attention scoring, GitHub activity, app store proxy
  - cross_asset_signals: correlations, dollar headwind, rate regime, momentum
"""

import math
import sys
import os
from typing import Dict, Optional
import pytest

# Ensure the parent research/alt_data package is importable when running
# tests directly from the tests/ directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from alt_data.macro_regime import (
    MacroRegime,
    MacroRegimeClassifier,
    MacroIndicator,
    Direction,
    YieldCurveMonitor,
    CreditSpreadMonitor,
    RegimeTransitionMatrix,
    _compute_z_score,
    _compute_direction,
)
from alt_data.on_chain_advanced import (
    RealizedPriceBands,
    ExchangeFlowAnalyzer,
    MinerSignal,
    StablecoinRatio,
    NVTSignal,
)
from alt_data.satellite_web import (
    GoogleTrendsProxy,
    GitHubActivitySignal,
    AppStoreProxy,
)
from alt_data.cross_asset_signals import (
    EquityCryptoCorrelation,
    DollarCycleSignal,
    RateImpact,
    RateRegime,
    CrossAssetMomentum,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _feed_classifier(
    classifier: MacroRegimeClassifier,
    indicators: dict,
    periods: int = 25,
) -> MacroRegime:
    """Feed the same indicator snapshot to a classifier for `periods` steps."""
    regime = MacroRegime.NEUTRAL
    for _ in range(periods):
        regime = classifier.classify(indicators)
    return regime


# ===========================================================================
# MacroIndicator tests
# ===========================================================================

class TestMacroIndicator:
    def test_basic_construction(self):
        ind = MacroIndicator(
            name="VIX",
            value=25.0,
            z_score=1.5,
            direction=Direction.UP,
            source="CBOE",
        )
        assert ind.name == "VIX"
        assert ind.value == 25.0
        assert ind.z_score == 1.5
        assert ind.direction == Direction.UP
        assert ind.source == "CBOE"

    def test_is_extreme_true(self):
        ind = MacroIndicator("VIX", 45.0, 2.5, Direction.UP, "CBOE")
        assert ind.is_extreme(2.0) is True

    def test_is_extreme_false(self):
        ind = MacroIndicator("VIX", 18.0, 0.3, Direction.FLAT, "CBOE")
        assert ind.is_extreme(2.0) is False

    def test_is_extreme_default_threshold(self):
        ind = MacroIndicator("VIX", 38.0, 1.9, Direction.UP, "CBOE")
        assert ind.is_extreme() is False
        ind2 = MacroIndicator("VIX", 38.0, 2.0, Direction.UP, "CBOE")
        assert ind2.is_extreme() is True

    def test_repr(self):
        ind = MacroIndicator("X", 1.0, 0.5, Direction.FLAT, "SRC")
        r = repr(ind)
        assert "MacroIndicator" in r
        assert "X" in r


# ===========================================================================
# _compute_z_score tests
# ===========================================================================

class TestComputeZScore:
    def test_basic_z_score(self):
        history = [10.0, 12.0, 11.0, 9.0, 13.0]
        # value at the mean should have z=0
        mean = sum(history) / len(history)
        z = _compute_z_score(mean, history)
        assert abs(z) < 1e-10

    def test_above_mean_positive_z(self):
        history = [1.0, 2.0, 3.0, 4.0, 5.0]
        z = _compute_z_score(10.0, history)
        assert z > 0

    def test_below_mean_negative_z(self):
        history = [1.0, 2.0, 3.0, 4.0, 5.0]
        z = _compute_z_score(-5.0, history)
        assert z < 0

    def test_short_history_returns_zero(self):
        assert _compute_z_score(5.0, []) == 0.0
        assert _compute_z_score(5.0, [3.0]) == 0.0

    def test_zero_variance_returns_zero(self):
        history = [5.0, 5.0, 5.0, 5.0]
        z = _compute_z_score(5.0, history)
        assert z == 0.0


# ===========================================================================
# MacroRegimeClassifier -- RISK_ON
# ===========================================================================

class TestMacroRegimeRiskOn:
    def test_macro_regime_risk_on(self):
        """Classic RISK_ON: calm VIX, positive equity and crypto momentum."""
        clf = MacroRegimeClassifier()
        indicators = {
            "vix":                 14.0,
            "yield_curve_2y10y":   0.003,
            "credit_spread_hy":    280.0,
            "inflation_yoy":       0.025,
            "gdp_growth_yoy":      0.025,
            "equity_momentum_20d": 0.04,
            "crypto_momentum_20d": 0.09,
        }
        regime = _feed_classifier(clf, indicators)
        assert regime == MacroRegime.RISK_ON

    def test_risk_on_crypto_momentum_required(self):
        """Without strong crypto momentum, RISK_ON should not trigger."""
        clf = MacroRegimeClassifier()
        indicators = {
            "vix":                 14.0,
            "yield_curve_2y10y":   0.003,
            "credit_spread_hy":    280.0,
            "inflation_yoy":       0.025,
            "gdp_growth_yoy":      0.025,
            "equity_momentum_20d": 0.04,
            "crypto_momentum_20d": 0.01,  # too low
        }
        regime = _feed_classifier(clf, indicators)
        assert regime != MacroRegime.RISK_ON

    def test_risk_on_allocation_multiplier(self):
        clf = MacroRegimeClassifier()
        assert clf.get_crypto_allocation_multiplier(MacroRegime.RISK_ON) == 1.2

    def test_goldilocks_allocation_multiplier(self):
        clf = MacroRegimeClassifier()
        assert clf.get_crypto_allocation_multiplier(MacroRegime.GOLDILOCKS) == 1.1

    def test_neutral_allocation_multiplier(self):
        clf = MacroRegimeClassifier()
        assert clf.get_crypto_allocation_multiplier(MacroRegime.NEUTRAL) == 1.0

    def test_risk_off_allocation_multiplier(self):
        clf = MacroRegimeClassifier()
        assert clf.get_crypto_allocation_multiplier(MacroRegime.RISK_OFF) == 0.5

    def test_stagflationary_allocation_multiplier(self):
        clf = MacroRegimeClassifier()
        assert clf.get_crypto_allocation_multiplier(MacroRegime.STAGFLATIONARY) == 0.7


# ===========================================================================
# MacroRegimeClassifier -- RISK_OFF
# ===========================================================================

class TestMacroRegimeRiskOff:
    def test_macro_regime_risk_off(self):
        """Classic RISK_OFF: VIX >30 and HY spreads >200bp."""
        clf = MacroRegimeClassifier()
        indicators = {
            "vix":               35.0,
            "credit_spread_hy":  450.0,
            "inflation_yoy":     0.03,
            "gdp_growth_yoy":    0.015,
        }
        regime = _feed_classifier(clf, indicators)
        assert regime == MacroRegime.RISK_OFF

    def test_risk_off_boundary_vix(self):
        """VIX exactly at threshold -- should not trigger RISK_OFF on its own."""
        clf = MacroRegimeClassifier()
        indicators = {
            "vix":              30.0,   # at threshold exactly
            "credit_spread_hy": 201.0,  # just above threshold
            "gdp_growth_yoy":   0.02,
        }
        regime = _feed_classifier(clf, indicators)
        # VIX must be ABOVE 30 (strictly), so 30.0 should not trigger
        assert regime != MacroRegime.RISK_OFF

    def test_risk_off_high_vix_tight_credit(self):
        """High VIX alone is not enough -- credit spreads must also be wide."""
        clf = MacroRegimeClassifier()
        indicators = {
            "vix":              35.0,
            "credit_spread_hy": 150.0,  # below threshold
        }
        regime = _feed_classifier(clf, indicators)
        assert regime != MacroRegime.RISK_OFF

    def test_risk_off_reduces_allocation(self):
        """RISK_OFF allocation multiplier should be the lowest among regimes."""
        clf = MacroRegimeClassifier()
        multipliers = [
            clf.get_crypto_allocation_multiplier(r) for r in MacroRegime
        ]
        ro_mult = clf.get_crypto_allocation_multiplier(MacroRegime.RISK_OFF)
        assert ro_mult == min(m for m in multipliers if m > 0)


# ===========================================================================
# MacroRegimeClassifier -- GOLDILOCKS and STAGFLATIONARY
# ===========================================================================

class TestMacroRegimeGoldilocksAndStagflation:
    def test_macro_regime_goldilocks(self):
        """Goldilocks: very low VIX, positive curve, decent growth."""
        clf = MacroRegimeClassifier()
        indicators = {
            "vix":               12.0,
            "yield_curve_2y10y": 0.008,  # 80bp positive slope
            "gdp_growth_yoy":    0.025,
            "inflation_yoy":     0.02,
            "credit_spread_hy":  250.0,
        }
        regime = _feed_classifier(clf, indicators)
        assert regime == MacroRegime.GOLDILOCKS

    def test_macro_regime_stagflationary(self):
        """Stagflation: high inflation + weak growth."""
        clf = MacroRegimeClassifier()
        indicators = {
            "vix":              22.0,
            "inflation_yoy":    0.06,   # 6% inflation
            "gdp_growth_yoy":   0.005,  # 0.5% growth
            "credit_spread_hy": 280.0,
        }
        regime = _feed_classifier(clf, indicators)
        assert regime == MacroRegime.STAGFLATIONARY

    def test_goldilocks_requires_low_vix(self):
        """GOLDILOCKS requires VIX < 15."""
        clf = MacroRegimeClassifier()
        indicators = {
            "vix":               16.0,  # too high for goldilocks
            "yield_curve_2y10y": 0.008,
            "gdp_growth_yoy":    0.025,
        }
        regime = _feed_classifier(clf, indicators)
        assert regime != MacroRegime.GOLDILOCKS

    def test_stagflation_requires_both_conditions(self):
        """Stagflation requires BOTH high inflation AND weak growth."""
        clf = MacroRegimeClassifier()
        # High inflation but decent growth -- should not be stagflationary
        indicators = {
            "vix":            22.0,
            "inflation_yoy":  0.06,
            "gdp_growth_yoy": 0.03,  # strong growth
        }
        regime = _feed_classifier(clf, indicators)
        assert regime != MacroRegime.STAGFLATIONARY


# ===========================================================================
# MacroRegimeClassifier -- majority vote smoothing
# ===========================================================================

class TestMacroRegimeMajorityVote:
    def test_majority_vote_smoothing_prevents_whipsaw(self):
        """
        If 19 RISK_ON signals are followed by 1 RISK_OFF, the classifier
        should still report RISK_ON due to majority vote.
        """
        clf = MacroRegimeClassifier(majority_window=20)
        risk_on_indicators = {
            "vix":                 14.0,
            "yield_curve_2y10y":   0.003,
            "credit_spread_hy":    280.0,
            "inflation_yoy":       0.025,
            "gdp_growth_yoy":      0.025,
            "equity_momentum_20d": 0.04,
            "crypto_momentum_20d": 0.09,
        }
        risk_off_indicators = {
            "vix":              35.0,
            "credit_spread_hy": 450.0,
        }
        # Feed 19 RISK_ON periods
        for _ in range(19):
            clf.classify(risk_on_indicators)
        # One RISK_OFF
        result = clf.classify(risk_off_indicators)
        # Majority should still be RISK_ON
        assert result == MacroRegime.RISK_ON

    def test_regime_resets_after_sustained_change(self):
        """After 25 consecutive RISK_OFF periods, the regime should shift."""
        clf = MacroRegimeClassifier(majority_window=20)
        risk_off = {
            "vix":              35.0,
            "credit_spread_hy": 450.0,
        }
        result = _feed_classifier(clf, risk_off, periods=25)
        assert result == MacroRegime.RISK_OFF


# ===========================================================================
# YieldCurveMonitor tests
# ===========================================================================

class TestYieldCurveInversionDetection:
    def test_yield_curve_inversion_detection(self):
        """2y > 10y -> is_inverted() should return True."""
        monitor = YieldCurveMonitor()
        monitor.update(y2=5.0, y5=4.8, y10=4.5, y30=4.6)
        assert monitor.is_inverted() is True

    def test_yield_curve_normal_not_inverted(self):
        monitor = YieldCurveMonitor()
        monitor.update(y2=3.5, y5=4.0, y10=4.5, y30=4.8)
        assert monitor.is_inverted() is False

    def test_yield_curve_slope(self):
        monitor = YieldCurveMonitor()
        monitor.update(y2=3.0, y5=3.5, y10=4.0, y30=4.5)
        assert abs(monitor.get_slope() - 1.0) < 1e-9

    def test_inversion_duration_increases(self):
        monitor = YieldCurveMonitor()
        for i in range(10):
            monitor.update(y2=5.0, y5=4.8, y10=4.5, y30=4.6)
        assert monitor.inversion_duration_days() == 10

    def test_inversion_duration_resets_on_normalization(self):
        monitor = YieldCurveMonitor()
        for _ in range(5):
            monitor.update(y2=5.0, y5=4.8, y10=4.5, y30=4.6)
        # Normalize curve
        monitor.update(y2=3.0, y5=3.5, y10=4.0, y30=4.5)
        assert monitor.inversion_duration_days() == 0

    def test_recession_warning_after_30_days(self):
        monitor = YieldCurveMonitor(recession_warning_days=30)
        for _ in range(30):
            monitor.update(y2=5.0, y5=4.8, y10=4.5, y30=4.6)
        assert monitor.is_recession_warning() is True

    def test_no_recession_warning_before_30_days(self):
        monitor = YieldCurveMonitor(recession_warning_days=30)
        for _ in range(29):
            monitor.update(y2=5.0, y5=4.8, y10=4.5, y30=4.6)
        assert monitor.is_recession_warning() is False

    def test_curve_regime_steep_normal(self):
        monitor = YieldCurveMonitor()
        monitor.update(y2=2.0, y5=3.0, y10=4.0, y30=4.5)
        assert monitor.get_curve_regime() == "STEEP_NORMAL"

    def test_curve_regime_deeply_inverted(self):
        monitor = YieldCurveMonitor()
        monitor.update(y2=6.0, y5=5.5, y10=4.8, y30=4.7)
        assert monitor.get_curve_regime() == "DEEPLY_INVERTED"

    def test_update_without_prior_raises_on_slope(self):
        monitor = YieldCurveMonitor()
        with pytest.raises(ValueError):
            monitor.get_slope()


# ===========================================================================
# CreditSpreadMonitor tests
# ===========================================================================

class TestCreditSpreadMonitor:
    def test_credit_spread_risk_signal_stress(self):
        """Very wide spreads should produce a negative risk signal."""
        monitor = CreditSpreadMonitor()
        # Feed 30 periods of tight spreads to build history
        for _ in range(30):
            monitor.update(ig_oas_bp=100.0, hy_oas_bp=300.0)
        # Then feed stress levels
        for _ in range(5):
            monitor.update(ig_oas_bp=300.0, hy_oas_bp=700.0)
        signal = monitor.risk_signal()
        assert signal < 0.0

    def test_credit_spread_risk_signal_easy(self):
        """Tight spreads should produce a positive risk signal."""
        monitor = CreditSpreadMonitor()
        # Feed wide spreads to build history
        for _ in range(30):
            monitor.update(ig_oas_bp=250.0, hy_oas_bp=600.0)
        # Then tight conditions
        for _ in range(5):
            monitor.update(ig_oas_bp=60.0, hy_oas_bp=200.0)
        signal = monitor.risk_signal()
        assert signal > 0.0

    def test_risk_signal_clamped(self):
        monitor = CreditSpreadMonitor()
        for _ in range(50):
            monitor.update(ig_oas_bp=80.0, hy_oas_bp=250.0)
        signal = monitor.risk_signal()
        assert -1.0 <= signal <= 1.0

    def test_is_credit_stress_true(self):
        monitor = CreditSpreadMonitor()
        monitor.update(ig_oas_bp=200.0, hy_oas_bp=650.0)
        assert monitor.is_credit_stress() is True

    def test_is_credit_easy_true(self):
        monitor = CreditSpreadMonitor()
        monitor.update(ig_oas_bp=80.0, hy_oas_bp=200.0)
        assert monitor.is_credit_easy() is True

    def test_blended_spread(self):
        monitor = CreditSpreadMonitor()
        monitor.update(ig_oas_bp=100.0, hy_oas_bp=400.0)
        blended = monitor.blended_spread_bp(ig_weight=0.3, hy_weight=0.7)
        expected = 0.3 * 100.0 + 0.7 * 400.0
        assert abs(blended - expected) < 1e-9

    def test_history_percentile_returns_valid_range(self):
        monitor = CreditSpreadMonitor()
        for i in range(30):
            monitor.update(ig_oas_bp=float(80 + i), hy_oas_bp=float(250 + i * 5))
        pct = monitor.history_percentile("hy")
        assert 0.0 <= pct <= 100.0


# ===========================================================================
# RegimeTransitionMatrix tests
# ===========================================================================

class TestRegimeTransitionMatrix:
    def test_transition_probability_sums_to_one(self):
        matrix = RegimeTransitionMatrix()
        regimes = [
            MacroRegime.RISK_ON, MacroRegime.RISK_ON,
            MacroRegime.NEUTRAL, MacroRegime.RISK_OFF,
            MacroRegime.NEUTRAL, MacroRegime.RISK_ON,
        ]
        for r in regimes:
            matrix.record(r)
        total = sum(
            matrix.transition_probability(MacroRegime.RISK_ON, r2)
            for r2 in MacroRegime
        )
        assert abs(total - 1.0) < 1e-9

    def test_stability_score_for_persistent_regime(self):
        matrix = RegimeTransitionMatrix()
        # NEUTRAL -> NEUTRAL -> NEUTRAL -> RISK_ON
        for _ in range(10):
            matrix.record(MacroRegime.NEUTRAL)
        matrix.record(MacroRegime.RISK_ON)
        # NEUTRAL self-transition should be high
        score = matrix.stability_score(MacroRegime.NEUTRAL)
        assert score > 0.8


# ===========================================================================
# RealizedPriceBands tests
# ===========================================================================

class TestRealizedPriceBands:
    def test_is_above_sth_rp_true(self):
        bands = RealizedPriceBands()
        bands.update(price=50000, sth_realized_price=40000, lth_realized_price=25000)
        assert bands.is_above_sth_rp(50000) is True

    def test_is_above_sth_rp_false(self):
        bands = RealizedPriceBands()
        bands.update(price=30000, sth_realized_price=40000, lth_realized_price=25000)
        assert bands.is_above_sth_rp(30000) is False

    def test_sth_profit_loss_ratio_positive(self):
        bands = RealizedPriceBands()
        bands.update(price=50000, sth_realized_price=40000, lth_realized_price=25000)
        ratio = bands.sth_profit_loss_ratio(50000)
        assert abs(ratio - 0.25) < 1e-9

    def test_sth_profit_loss_ratio_negative(self):
        bands = RealizedPriceBands()
        bands.update(price=30000, sth_realized_price=40000, lth_realized_price=20000)
        ratio = bands.sth_profit_loss_ratio(30000)
        assert ratio < 0

    def test_realized_price_ratio(self):
        bands = RealizedPriceBands()
        bands.update(price=50000, sth_realized_price=45000, lth_realized_price=30000)
        rpr = bands.realized_price_ratio()
        assert abs(rpr - 45000 / 30000) < 1e-9

    def test_composite_signal_above_both_bands(self):
        """Price well above both bands -> positive composite signal."""
        bands = RealizedPriceBands()
        bands.update(price=100000, sth_realized_price=40000, lth_realized_price=25000)
        sig = bands.composite_signal(100000)
        assert sig > 0

    def test_composite_signal_below_both_bands(self):
        """Price below both bands -> negative composite signal."""
        bands = RealizedPriceBands()
        bands.update(price=100000, sth_realized_price=40000, lth_realized_price=30000)
        sig = bands.composite_signal(10000)
        assert sig < 0

    def test_no_data_raises(self):
        bands = RealizedPriceBands()
        with pytest.raises(ValueError):
            bands.is_above_sth_rp(50000)

    def test_to_dict_keys(self):
        bands = RealizedPriceBands()
        bands.update(price=50000, sth_realized_price=40000, lth_realized_price=25000)
        d = bands.to_dict(50000)
        assert "sth_rp" in d
        assert "lth_rp" in d
        assert "composite_signal" in d


# ===========================================================================
# ExchangeFlowAnalyzer tests
# ===========================================================================

class TestExchangeFlowAccumulationSignal:
    def test_exchange_flow_accumulation_signal_positive(self):
        """Sustained outflows -> accumulation signal should be high."""
        analyzer = ExchangeFlowAnalyzer(short_window=7, long_window=30)
        # Feed 40 days of heavy outflows (more coins leaving exchanges)
        for _ in range(40):
            analyzer.update(inflow_btc=100.0, outflow_btc=500.0)
        signal = analyzer.accumulation_signal()
        assert signal > 0.5

    def test_exchange_flow_distribution_signal_positive(self):
        """Sustained inflows -> distribution signal should be high."""
        analyzer = ExchangeFlowAnalyzer(short_window=7, long_window=30)
        # Feed baseline
        for _ in range(30):
            analyzer.update(inflow_btc=300.0, outflow_btc=300.0)
        # Then spike of inflows
        for _ in range(10):
            analyzer.update(inflow_btc=1000.0, outflow_btc=100.0)
        signal = analyzer.distribution_signal()
        assert signal > 0.3

    def test_composite_flow_signal_range(self):
        analyzer = ExchangeFlowAnalyzer()
        for _ in range(50):
            analyzer.update(inflow_btc=200.0, outflow_btc=400.0)
        signal = analyzer.composite_flow_signal()
        assert -1.0 <= signal <= 1.0

    def test_neutral_flows_near_zero(self):
        """Balanced flows should produce near-zero signals."""
        analyzer = ExchangeFlowAnalyzer()
        for _ in range(40):
            analyzer.update(inflow_btc=300.0, outflow_btc=300.0)
        dist = analyzer.distribution_signal()
        acc  = analyzer.accumulation_signal()
        # Both should be low since no net imbalance
        assert dist < 0.3
        assert acc  < 0.3

    def test_exchange_balance_trend_accumulating(self):
        analyzer = ExchangeFlowAnalyzer(short_window=5, long_window=20)
        for _ in range(25):
            analyzer.update(inflow_btc=100.0, outflow_btc=600.0)
        trend = analyzer.exchange_balance_trend()
        assert trend == "ACCUMULATING"

    def test_rolling_net_flow_sum(self):
        analyzer = ExchangeFlowAnalyzer()
        for i in range(10):
            analyzer.update(inflow_btc=float(100 + i), outflow_btc=100.0)
        # Last 5 periods: net flows 5,6,7,8,9
        rolling = analyzer.rolling_net_flow(5)
        expected = sum(range(5, 10))
        assert abs(rolling - expected) < 1e-9


# ===========================================================================
# MinerSignal tests
# ===========================================================================

class TestMinerSignal:
    def test_miner_signal_capitulation(self):
        """Low revenue/cost ratio -> is_capitulating() should be True."""
        miner = MinerSignal()
        miner.update(
            miner_revenue_usd=10_000_000,
            miner_cost_usd=12_000_000,  # cost > revenue
            hashrate_eh_s=400.0,
        )
        assert miner.is_capitulating() is True

    def test_miner_signal_profitable(self):
        """High revenue/cost ratio -> not capitulating."""
        miner = MinerSignal()
        miner.update(
            miner_revenue_usd=30_000_000,
            miner_cost_usd=15_000_000,
            hashrate_eh_s=600.0,
        )
        assert miner.is_capitulating() is False

    def test_get_miner_signal_range(self):
        """Miner signal should always be in [-1, +1]."""
        miner = MinerSignal()
        for i in range(70):
            miner.update(
                miner_revenue_usd=float(20_000_000 + i * 100_000),
                miner_cost_usd=15_000_000.0,
                hashrate_eh_s=float(400 + i),
                miner_outflow_btc=float(200 + i * 2),
            )
        signal = miner.get_miner_signal()
        assert -1.0 <= signal <= 1.0

    def test_hash_ribbon_signal_positive_when_recovering(self):
        """When 30d SMA > 60d SMA, hash ribbon should be positive."""
        miner = MinerSignal()
        # Feed 60 periods of growing hashrate
        for i in range(65):
            miner.update(
                miner_revenue_usd=20_000_000.0,
                miner_cost_usd=15_000_000.0,
                hashrate_eh_s=float(400 + i * 5),  # steadily increasing
            )
        ribbon = miner.hash_ribbon_signal()
        assert ribbon > 0.0

    def test_days_in_capitulation_counter(self):
        miner = MinerSignal()
        for _ in range(7):
            miner.update(
                miner_revenue_usd=10_000_000.0,
                miner_cost_usd=12_000_000.0,
                hashrate_eh_s=400.0,
            )
        assert miner.days_in_capitulation() == 7

    def test_days_in_capitulation_resets(self):
        miner = MinerSignal()
        for _ in range(5):
            miner.update(10_000_000.0, 12_000_000.0, 400.0)
        # Profitable period
        miner.update(25_000_000.0, 15_000_000.0, 400.0)
        assert miner.days_in_capitulation() == 0


# ===========================================================================
# StablecoinRatio tests
# ===========================================================================

class TestStablecoinRatio:
    def test_liquidity_signal_high_ssr(self):
        """High SSR percentile -> positive liquidity signal."""
        sr = StablecoinRatio()
        # Build history with low SSR
        for _ in range(50):
            sr.update(stablecoin_market_cap_usd=100e9, total_crypto_market_cap_usd=2000e9)
        # Then high SSR (lots of stablecoins relative to crypto)
        for _ in range(5):
            sr.update(stablecoin_market_cap_usd=400e9, total_crypto_market_cap_usd=2000e9)
        signal = sr.get_liquidity_signal()
        assert signal > 0.0

    def test_liquidity_signal_low_ssr(self):
        """Low SSR percentile -> negative liquidity signal."""
        sr = StablecoinRatio()
        # Build history with high SSR
        for _ in range(50):
            sr.update(stablecoin_market_cap_usd=400e9, total_crypto_market_cap_usd=2000e9)
        # Then low SSR
        for _ in range(5):
            sr.update(stablecoin_market_cap_usd=50e9, total_crypto_market_cap_usd=2000e9)
        signal = sr.get_liquidity_signal()
        assert signal < 0.0

    def test_liquidity_signal_clamped(self):
        sr = StablecoinRatio()
        for _ in range(30):
            sr.update(stablecoin_market_cap_usd=200e9, total_crypto_market_cap_usd=2000e9)
        signal = sr.get_liquidity_signal()
        assert -1.0 <= signal <= 1.0

    def test_dry_powder(self):
        sr = StablecoinRatio()
        sr.update(stablecoin_market_cap_usd=250e9, total_crypto_market_cap_usd=1500e9)
        assert abs(sr.dry_powder_usd() - 250e9) < 1.0

    def test_ssr_percentile_range(self):
        sr = StablecoinRatio()
        for i in range(30):
            sr.update(float(100e9 + i * 5e9), 2000e9)
        pct = sr.ssr_percentile()
        assert 0.0 <= pct <= 100.0


# ===========================================================================
# NVTSignal tests
# ===========================================================================

class TestNVTSignal:
    def test_nvt_signal_bounds(self):
        """NVT normalized signal should be in [-1, +1]."""
        nvt = NVTSignal()
        for i in range(120):
            market_cap    = float(400e9 + i * 1e9)
            daily_vol     = float(5e9 + i * 50e6)
            norm = nvt.nvt_signal_normalized(market_cap, daily_vol)
            assert -1.0 <= norm <= 1.0

    def test_nvt_basic_computation(self):
        """NVT = market_cap / smoothed_tx_volume."""
        nvt = NVTSignal()
        # Feed 90 identical periods for a stable moving average
        for _ in range(90):
            raw = nvt.nvt_signal(market_cap=900e9, daily_volume=10e9, window=90)
        # After 90 identical periods, smoothed vol == 10e9, NVT == 90
        assert abs(raw - 90.0) < 1.0

    def test_nvt_high_nvt_overvalued(self):
        """High NVT relative to history -> is_overvalued should return True."""
        nvt = NVTSignal()
        # Build history with low NVT
        for _ in range(100):
            nvt.nvt_signal(market_cap=100e9, daily_volume=10e9, window=50)
        # Then spike market cap massively (high NVT)
        for _ in range(5):
            nvt.nvt_signal(market_cap=5000e9, daily_volume=10e9, window=50)
        assert nvt.is_overvalued() is True

    def test_nvt_low_nvt_undervalued(self):
        """Low NVT relative to history -> is_undervalued should return True."""
        nvt = NVTSignal()
        # Build history with high NVT
        for _ in range(100):
            nvt.nvt_signal(market_cap=5000e9, daily_volume=10e9, window=50)
        # Then collapse market cap (low NVT)
        for _ in range(5):
            nvt.nvt_signal(market_cap=100e9, daily_volume=10e9, window=50)
        assert nvt.is_undervalued() is True

    def test_nvt_no_history_returns_50th_percentile(self):
        nvt = NVTSignal()
        pct = nvt.nvt_percentile()
        assert pct == 50.0

    def test_nvt_zero_volume_returns_nan(self):
        nvt = NVTSignal()
        result = nvt.nvt_signal(market_cap=100e9, daily_volume=0.0, window=90)
        assert math.isnan(result)


# ===========================================================================
# GoogleTrendsProxy tests
# ===========================================================================

class TestGoogleTrendsProxy:
    def test_attention_score_spike(self):
        """When headline frequency jumps 3x, attention score should be positive."""
        proxy = GoogleTrendsProxy(baseline_window_days=10, spike_threshold=3.0)
        # Build baseline: 2 mentions per day
        for _ in range(15):
            proxy.compute_attention_score(
                ["bitcoin price rises", "altcoin rally today"], "bitcoin", window_days=5
            )
        # Spike: 10 mentions in a single day
        headlines_spike = ["bitcoin" for _ in range(10)]
        score = proxy.compute_attention_score(headlines_spike, "bitcoin", window_days=3)
        assert score > 0.0

    def test_attention_score_fade(self):
        """When headline frequency drops well below baseline, score should be negative."""
        proxy = GoogleTrendsProxy(baseline_window_days=10, spike_threshold=3.0)
        # Build high baseline
        for _ in range(15):
            proxy.compute_attention_score(
                ["bitcoin bitcoin bitcoin bitcoin bitcoin"] * 5, "bitcoin", window_days=5
            )
        # Fade: zero mentions
        score = proxy.compute_attention_score([], "bitcoin", window_days=5)
        assert score <= 0.0

    def test_attention_score_neutral(self):
        """At baseline level, attention score should be near 0."""
        proxy = GoogleTrendsProxy()
        for _ in range(20):
            proxy.compute_attention_score(
                ["bitcoin news update"], "bitcoin", window_days=7
            )
        score = proxy.compute_attention_score(["bitcoin news update"], "bitcoin", window_days=7)
        assert -0.3 <= score <= 0.3

    def test_is_spiking_true(self):
        proxy = GoogleTrendsProxy(baseline_window_days=10, spike_threshold=3.0)
        # Low baseline (1 mention/day)
        for _ in range(15):
            proxy.compute_attention_score(["unrelated news"], "bitcoin", window_days=5)
        # Spike
        proxy.compute_attention_score(["bitcoin bitcoin bitcoin bitcoin"] * 5, "bitcoin")
        assert proxy.is_spiking("bitcoin") is True

    def test_multi_keyword_score_range(self):
        proxy = GoogleTrendsProxy()
        headlines = ["bitcoin rally", "ethereum bullish", "crypto market up"]
        for _ in range(15):
            proxy.compute_attention_score(headlines, "bitcoin")
            proxy.compute_attention_score(headlines, "ethereum")
        score = proxy.multi_keyword_score(headlines, ["bitcoin", "ethereum"])
        assert -1.0 <= score <= 1.0


# ===========================================================================
# GitHubActivitySignal tests
# ===========================================================================

class TestGitHubActivitySignal:
    def test_dev_activity_score_range(self):
        """compute_dev_activity should return value in [0, 1]."""
        gh = GitHubActivitySignal()
        stats = {"commits_30d": 150, "stars_total": 5000, "forks_total": 800, "open_prs": 45}
        score = gh.compute_dev_activity(stats)
        assert 0.0 <= score <= 1.0

    def test_dev_activity_high_commits(self):
        """Very active repo should score near 1."""
        gh = GitHubActivitySignal()
        stats = {"commits_30d": 500, "stars_7d_growth": 5000, "forks_7d_growth": 1000, "open_prs": 200}
        score = gh.compute_dev_activity(stats)
        assert score > 0.5

    def test_dev_activity_zero_activity(self):
        """Inactive repo should score near 0."""
        gh = GitHubActivitySignal()
        stats = {"commits_30d": 0, "stars_total": 0, "forks_total": 0, "open_prs": 0}
        score = gh.compute_dev_activity(stats)
        assert score == 0.0

    def test_ecosystem_signal_aggregation(self):
        """Ecosystem signal should average across repos."""
        gh = GitHubActivitySignal()
        repos = ["bitcoin/bitcoin", "ethereum/go-ethereum"]
        stats_map = {
            "bitcoin/bitcoin":     {"commits_30d": 100, "stars_total": 10000},
            "ethereum/go-ethereum": {"commits_30d": 200, "stars_total": 8000},
        }
        score = gh.ecosystem_signal(repos, stats_map)
        assert 0.0 <= score <= 1.0

    def test_is_trending_up_true(self):
        gh = GitHubActivitySignal()
        repo = "test/repo"
        # Feed growing commits
        for i in range(60):
            gh.record(repo, {"commits_30d": float(50 + i * 3)})
        assert gh.is_trending_up(repo, window=14) is True

    def test_is_trending_up_false(self):
        gh = GitHubActivitySignal()
        repo = "test/repo"
        # Feed declining commits
        for i in range(60):
            gh.record(repo, {"commits_30d": float(200 - i * 2)})
        assert gh.is_trending_up(repo, window=14) is False


# ===========================================================================
# AppStoreProxy tests
# ===========================================================================

class TestAppStoreProxy:
    def test_user_growth_signal_positive(self):
        """Growth-heavy headlines should produce a positive signal."""
        proxy = AppStoreProxy()
        growth_headlines = [
            "Coinbase records new user sign-ups milestone",
            "Exchange reports record downloads this quarter",
            "New accounts surged 50% following Bitcoin rally",
        ] * 5
        # Build baseline with neutral headlines
        for _ in range(15):
            proxy.user_growth_signal(["crypto market update today"])
        signal = proxy.user_growth_signal(growth_headlines)
        assert signal > 0.0

    def test_user_growth_signal_range(self):
        """Signal must always be in [-1, +1]."""
        proxy = AppStoreProxy()
        headlines = ["bitcoin users leaving platform", "crypto withdrawals halted record"]
        for _ in range(30):
            proxy.user_growth_signal(["neutral news"])
        signal = proxy.user_growth_signal(headlines)
        assert -1.0 <= signal <= 1.0

    def test_decline_alert_triggers(self):
        proxy = AppStoreProxy()
        # Low decline baseline
        for _ in range(20):
            proxy.user_growth_signal(["normal crypto news today"])
        # Spike of decline signals
        for _ in range(3):
            proxy.user_growth_signal(
                ["users leaving exchange en masse", "accounts deleted after hack",
                 "user exodus from platform", "crypto withdrawals halted"] * 3
            )
        assert proxy.decline_alert() is True

    def test_combined_sentiment_keys(self):
        proxy = AppStoreProxy()
        result = proxy.combined_sentiment_score(["bitcoin new users record downloads"])
        assert "user_growth_signal" in result
        assert "growth_mentions" in result
        assert "decline_mentions" in result


# ===========================================================================
# EquityCryptoCorrelation tests
# ===========================================================================

class TestEquityCryptoCorrelation:
    def _feed_positively_correlated(self, corr: EquityCryptoCorrelation, n: int = 40):
        """Feed price series where BTC and SPX move together."""
        import random
        random.seed(42)
        btc = 50000.0
        spx = 4500.0
        ndx = 15000.0
        for _ in range(n):
            shock = random.gauss(0, 0.02)
            btc  *= (1 + shock)
            spx  *= (1 + shock + random.gauss(0, 0.005))
            ndx  *= (1 + shock + random.gauss(0, 0.005))
            corr.update(btc, {"SPX": spx, "NDX": ndx, "GLD": 1800.0, "DXY": 104.0, "TLT": 95.0})

    def test_risk_on_signal_positive_when_correlated(self):
        corr = EquityCryptoCorrelation(window=20)
        self._feed_positively_correlated(corr)
        signal = corr.risk_on_signal()
        assert signal > 0.0

    def test_safe_haven_signal_range(self):
        corr = EquityCryptoCorrelation(window=20)
        self._feed_positively_correlated(corr)
        signal = corr.safe_haven_signal()
        assert 0.0 <= signal <= 1.0

    def test_all_correlations_keys(self):
        corr = EquityCryptoCorrelation()
        self._feed_positively_correlated(corr)
        result = corr.all_correlations()
        assert set(result.keys()) == {"SPX", "NDX", "GLD", "DXY", "TLT"}

    def test_regime_classification_returns_string(self):
        corr = EquityCryptoCorrelation(window=20)
        self._feed_positively_correlated(corr)
        regime = corr.regime_classification()
        assert regime in ("RISK_ASSET", "SAFE_HAVEN", "DECORRELATED", "DOLLAR_DRIVEN")

    def test_correlation_insufficient_data_returns_zero(self):
        corr = EquityCryptoCorrelation()
        assert corr.get_correlation("SPX") == 0.0


# ===========================================================================
# DollarCycleSignal tests
# ===========================================================================

class TestDollarHeadwind:
    def test_dollar_headwind_sign_strengthening_dxy(self):
        """Strengthening dollar should produce negative (headwind) signal for crypto."""
        signal = DollarCycleSignal()
        headwind = signal.get_dollar_headwind(dxy_momentum=0.05)  # 5% DXY rally
        assert headwind < 0.0

    def test_dollar_headwind_sign_weakening_dxy(self):
        """Weakening dollar should produce positive (tailwind) signal for crypto."""
        signal = DollarCycleSignal()
        headwind = signal.get_dollar_headwind(dxy_momentum=-0.05)  # 5% DXY decline
        assert headwind > 0.0

    def test_dollar_headwind_clamped(self):
        signal = DollarCycleSignal()
        # Extreme DXY surge
        headwind = signal.get_dollar_headwind(dxy_momentum=0.50)
        assert headwind == -1.0

    def test_dollar_headwind_from_history_strengthening(self):
        signal = DollarCycleSignal()
        # Feed rising DXY
        for i in range(60):
            signal.update(dxy_level=float(100 + i * 0.2))
        headwind = signal.get_dollar_headwind_from_history()
        assert headwind < 0.0

    def test_dollar_headwind_from_history_weakening(self):
        signal = DollarCycleSignal()
        # Feed falling DXY
        for i in range(60):
            signal.update(dxy_level=float(110 - i * 0.2))
        headwind = signal.get_dollar_headwind_from_history()
        assert headwind > 0.0

    def test_dxy_trend_strengthening(self):
        signal = DollarCycleSignal(short_window=5)
        for i in range(20):
            signal.update(float(100 + i * 0.5))
        assert signal.dxy_trend() == "STRENGTHENING"

    def test_dxy_trend_weakening(self):
        signal = DollarCycleSignal(short_window=5)
        for i in range(20):
            signal.update(float(110 - i * 0.5))
        assert signal.dxy_trend() == "WEAKENING"

    def test_dxy_at_extreme_above_110(self):
        signal = DollarCycleSignal()
        signal.update(112.0)
        assert signal.dxy_at_extreme() is True

    def test_dxy_at_extreme_normal(self):
        signal = DollarCycleSignal()
        signal.update(104.0)
        assert signal.dxy_at_extreme() is False


# ===========================================================================
# RateImpact tests
# ===========================================================================

class TestRateImpact:
    def test_rate_regime_hiking(self):
        rate = RateImpact()
        for r in [2.25, 2.50, 2.75, 3.00, 3.25]:
            rate.update(r)
        assert rate.get_rate_regime() == "HIKING"

    def test_rate_regime_cutting(self):
        rate = RateImpact()
        for r in [5.25, 5.00, 4.75, 4.50, 4.25]:
            rate.update(r)
        assert rate.get_rate_regime() == "CUTTING"

    def test_rate_regime_paused(self):
        rate = RateImpact()
        for r in [5.25, 5.25, 5.25, 5.25, 5.25]:
            rate.update(r)
        assert rate.get_rate_regime() == "PAUSED"

    def test_crypto_impact_cutting_positive(self):
        rate = RateImpact()
        for r in [5.25, 5.00, 4.75, 4.50, 4.25]:
            rate.update(r)
        score = rate.crypto_impact_score()
        assert score > 0.0

    def test_crypto_impact_hiking_negative(self):
        rate = RateImpact()
        for r in [2.25, 2.50, 2.75, 3.00, 3.25]:
            rate.update(r)
        score = rate.crypto_impact_score()
        assert score < 0.0

    def test_is_restrictive_true(self):
        rate = RateImpact(neutral_rate_est=2.5)
        rate.update(5.25)
        assert rate.is_restrictive() is True

    def test_is_restrictive_false(self):
        rate = RateImpact(neutral_rate_est=2.5)
        rate.update(1.0)
        assert rate.is_restrictive() is False

    def test_hike_cycle_duration(self):
        rate = RateImpact()
        for r in [2.0, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50]:
            rate.update(r)
        assert rate.hike_cycle_duration() > 0

    def test_no_data_raises(self):
        rate = RateImpact()
        with pytest.raises(ValueError):
            rate.current_rate()


# ===========================================================================
# CrossAssetMomentum tests
# ===========================================================================

class TestCrossAssetMomentum:
    def _build_price_series(
        self,
        mom: CrossAssetMomentum,
        n: int = 30,
        btc_trend: float = 0.01,
        spx_trend: float = 0.005,
    ) -> Dict[str, float]:
        """
        Feed price data with configurable trends per asset.
        Returns the final price dict so tests can pass it to get_cross_asset_signal.
        """
        prices = {"SPX": 4500.0, "NDX": 15000.0, "GLD": 1800.0, "TLT": 95.0, "BTC": 50000.0}
        trends = {
            "SPX": spx_trend,
            "NDX": spx_trend * 1.2,
            "GLD": 0.002,
            "TLT": -0.003,
            "BTC": btc_trend,
        }
        for _ in range(n):
            mom.update(prices)
            prices = {k: v * (1 + trends[k]) for k, v in prices.items()}
        return prices

    def test_cross_asset_signal_keys(self):
        mom = CrossAssetMomentum()
        final_prices = self._build_price_series(mom, n=25)
        result = mom.get_cross_asset_signal(final_prices)
        assert "BTC_in_portfolio" in result
        assert "portfolio_score" in result

    def test_cross_asset_momentum_weights_sum_to_one(self):
        """Portfolio weights for top_k assets should sum to 1.0."""
        mom = CrossAssetMomentum(top_k=3)
        final_prices = self._build_price_series(mom, n=25)
        result = mom.get_cross_asset_signal(final_prices)
        total_weight = sum(
            result[f"{asset}_portfolio_wt"] for asset in mom.assets
        )
        assert abs(total_weight - 1.0) < 1e-9

    def test_btc_in_portfolio_when_top_momentum(self):
        """When BTC has highest momentum, it should be in the portfolio."""
        mom = CrossAssetMomentum(top_k=3, window=20)
        # BTC outperforms all others -- use the final prices from the series
        final_prices = self._build_price_series(mom, n=25, btc_trend=0.05, spx_trend=0.001)
        result = mom.get_cross_asset_signal(final_prices)
        assert result["BTC_in_portfolio"] == 1.0

    def test_risk_parity_weights_sum_to_one(self):
        mom = CrossAssetMomentum()
        self._build_price_series(mom, n=30)
        weights = mom.risk_parity_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_btc_relative_momentum_positive_when_outperforming(self):
        mom = CrossAssetMomentum(window=15)
        final_prices = self._build_price_series(mom, n=30, btc_trend=0.03, spx_trend=0.001)
        mom.update(final_prices)  # ensure history includes final tick
        rel_mom = mom.btc_relative_momentum()
        assert rel_mom > 0.0

    def test_momentum_dispersion_positive(self):
        mom = CrossAssetMomentum(window=10)
        # GLD declining, BTC surging -> high dispersion
        self._build_price_series(mom, n=25, btc_trend=0.04, spx_trend=-0.01)
        dispersion = mom.momentum_dispersion()
        assert dispersion >= 0.0

    def test_correlation_matrix_diagonal_ones(self):
        mom = CrossAssetMomentum(window=15)
        self._build_price_series(mom, n=25)
        corr = mom.correlation_matrix()
        for asset in mom.assets:
            assert corr[(asset, asset)] == 1.0

    def test_correlation_matrix_symmetric(self):
        mom = CrossAssetMomentum(window=15)
        self._build_price_series(mom, n=25)
        corr = mom.correlation_matrix()
        for a in mom.assets:
            for b in mom.assets:
                assert abs(corr[(a, b)] - corr[(b, a)]) < 1e-9


# ===========================================================================
# Integration smoke test
# ===========================================================================

class TestIntegrationSmoke:
    def test_full_pipeline_no_crash(self):
        """
        Exercise the full alt_data pipeline end-to-end with synthetic data.
        Verifies no exceptions are raised and all signals are in valid ranges.
        """
        import random
        random.seed(123)

        # -- Macro regime
        clf = MacroRegimeClassifier()
        for _ in range(30):
            regime = clf.classify({
                "vix":                 random.uniform(12, 40),
                "credit_spread_hy":    random.uniform(200, 600),
                "inflation_yoy":       random.uniform(0.01, 0.07),
                "gdp_growth_yoy":      random.uniform(-0.01, 0.04),
                "equity_momentum_20d": random.uniform(-0.05, 0.05),
                "crypto_momentum_20d": random.uniform(-0.1, 0.15),
            })
        multiplier = clf.get_crypto_allocation_multiplier(regime)
        assert 0 < multiplier <= 1.5

        # -- On-chain signals
        bands = RealizedPriceBands()
        exc   = ExchangeFlowAnalyzer()
        miner = MinerSignal()
        sr    = StablecoinRatio()
        nvt   = NVTSignal()

        for i in range(50):
            p = 45000 + i * 300
            bands.update(p, sth_realized_price=40000.0, lth_realized_price=22000.0)
            exc.update(inflow_btc=random.uniform(100, 800), outflow_btc=random.uniform(100, 600))
            miner.update(
                miner_revenue_usd=random.uniform(15e6, 35e6),
                miner_cost_usd=15e6,
                hashrate_eh_s=random.uniform(400, 650),
                miner_outflow_btc=random.uniform(100, 400),
            )
            sr.update(stablecoin_market_cap_usd=200e9, total_crypto_market_cap_usd=2000e9)
            nvt.nvt_signal(market_cap=900e9, daily_volume=8e9 + i * 1e8, window=50)

        assert -1.0 <= exc.composite_flow_signal() <= 1.0
        assert -1.0 <= miner.get_miner_signal() <= 1.0
        assert -1.0 <= sr.get_liquidity_signal() <= 1.0

        # -- Cross-asset
        mom = CrossAssetMomentum()
        prices = {"SPX": 4500.0, "NDX": 15000.0, "GLD": 1800.0, "TLT": 95.0, "BTC": 50000.0}
        for _ in range(30):
            mom.update(prices)
            prices = {k: v * (1 + random.gauss(0.001, 0.01)) for k, v in prices.items()}
        result = mom.get_cross_asset_signal(prices)
        total_wt = sum(result[f"{a}_portfolio_wt"] for a in mom.assets)
        assert abs(total_wt - 1.0) < 1e-9
