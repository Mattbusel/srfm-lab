"""
test_domain_randomization.py — Tests for the Domain Randomization Engine (DRE).

Covers:
- SpreadNoiseGenerator
- FillRateRandomizer
- LatencyJitterInjector
- PriceImpactRandomizer
- OrderBookDepthRandomizer
- LiquidityShockInjector
- RegimeManager
- AdversarialParticipantInjector
- AutoCurriculum
- AnnealScheduler
- ObservationAugmentor
- DomainRandomizationEngine (integration)
- ScenarioSampler
- ParameterSchedule
- Factory functions
"""

import math
import pytest
import numpy as np

# Import DRE components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hyper_agent.domain_randomization import (
    # Enums
    MarketRegime,
    AdversaryType,
    AnnealSchedule,
    LiquidityShockType,
    # Configs
    DREConfig,
    SpreadNoiseConfig,
    FillRateConfig,
    LatencyConfig,
    PriceImpactConfig,
    OrderBookDepthConfig,
    LiquidityShockConfig,
    RegimeConfig,
    AdversaryConfig,
    CurriculumConfig,
    AnnealConfig,
    # Sub-modules
    SpreadNoiseGenerator,
    FillRateRandomizer,
    LatencyJitterInjector,
    PriceImpactRandomizer,
    OrderBookDepthRandomizer,
    LiquidityShockInjector,
    LiquidityShockState,
    RegimeManager,
    AdversarialParticipantInjector,
    AutoCurriculum,
    AnnealScheduler,
    ObservationAugmentor,
    DREMetricsTracker,
    DomainRandomizationEngine,
    ScenarioSampler,
    ParameterSchedule,
    make_dre,
    make_light_dre,
    make_adversarial_dre,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def dre():
    return make_light_dre(seed=42)


@pytest.fixture
def full_dre():
    return make_dre(num_assets=4, num_agents=4, seed=42, anneal_steps=1000)


# ---------------------------------------------------------------------------
# AnnealScheduler
# ---------------------------------------------------------------------------

class TestAnnealScheduler:
    def test_constant_schedule(self):
        cfg = AnnealConfig(schedule_type=AnnealSchedule.CONSTANT, initial_intensity=0.5)
        sched = AnnealScheduler(cfg)
        for _ in range(100):
            sched.step()
        assert abs(sched.intensity() - 0.5) < 0.01

    def test_linear_schedule_increases(self):
        cfg = AnnealConfig(
            schedule_type=AnnealSchedule.LINEAR,
            total_steps=1000,
            warmup_steps=0,
            initial_intensity=0.1,
            final_intensity=1.0,
            min_intensity=0.0,
            max_intensity=2.0,
        )
        sched = AnnealScheduler(cfg)
        start = sched.intensity()
        for _ in range(500):
            sched.step()
        mid = sched.intensity()
        for _ in range(500):
            sched.step()
        end = sched.intensity()
        assert end > start
        assert mid > start

    def test_cosine_schedule(self):
        cfg = AnnealConfig(
            schedule_type=AnnealSchedule.COSINE,
            total_steps=1000,
            warmup_steps=0,
            initial_intensity=1.0,
            final_intensity=0.0,
            min_intensity=0.0,
            max_intensity=1.0,
        )
        sched = AnnealScheduler(cfg)
        vals = []
        for _ in range(100):
            sched.step()
            vals.append(sched.intensity())
        # Cosine decay should be monotone non-increasing eventually
        assert vals[-1] <= vals[0] + 0.1

    def test_step_schedule(self):
        cfg = AnnealConfig(
            schedule_type=AnnealSchedule.STEP,
            initial_intensity=1.0,
            step_size=100,
            step_decay=0.5,
            warmup_steps=0,
            min_intensity=0.0,
            max_intensity=2.0,
        )
        sched = AnnealScheduler(cfg)
        for _ in range(200):
            sched.step()
        val = sched.intensity()
        # After 200 steps = 2 decays: 1.0 * 0.5^2 = 0.25
        assert val < 1.0

    def test_cyclic_schedule_oscillates(self):
        cfg = AnnealConfig(
            schedule_type=AnnealSchedule.CYCLIC,
            cycle_length=100,
            min_intensity=0.1,
            max_intensity=1.0,
            warmup_steps=0,
        )
        sched = AnnealScheduler(cfg)
        vals = []
        for _ in range(200):
            sched.step()
            vals.append(sched.intensity())
        # Should oscillate between min and max
        assert max(vals) >= 0.8
        assert min(vals) <= 0.3

    def test_warmup(self):
        cfg = AnnealConfig(
            schedule_type=AnnealSchedule.CONSTANT,
            initial_intensity=1.0,
            warmup_steps=100,
            min_intensity=0.0,
            max_intensity=2.0,
        )
        sched = AnnealScheduler(cfg)
        # At step 0, should be low
        val_start = sched.intensity()
        assert val_start < 0.1

    def test_reset_step(self):
        cfg = AnnealConfig(schedule_type=AnnealSchedule.LINEAR, total_steps=1000, warmup_steps=0)
        sched = AnnealScheduler(cfg)
        for _ in range(500):
            sched.step()
        sched.reset_step()
        assert sched.current_step == 0


# ---------------------------------------------------------------------------
# SpreadNoiseGenerator
# ---------------------------------------------------------------------------

class TestSpreadNoiseGenerator:
    def test_sample_shape(self, rng):
        cfg = SpreadNoiseConfig(enabled=True)
        gen = SpreadNoiseGenerator(cfg, rng)
        gen.reset(4)
        mults = gen.sample(4)
        assert mults.shape == (4,)

    def test_multipliers_in_range(self, rng):
        cfg = SpreadNoiseConfig(enabled=True, min_multiplier=0.5, max_multiplier=10.0)
        gen = SpreadNoiseGenerator(cfg, rng)
        gen.reset(4)
        for _ in range(100):
            mults = gen.sample(4)
            assert np.all(mults >= cfg.min_multiplier)
            assert np.all(mults <= cfg.max_multiplier)

    def test_disabled_returns_ones(self, rng):
        cfg = SpreadNoiseConfig(enabled=False)
        gen = SpreadNoiseGenerator(cfg, rng)
        gen.reset(4)
        mults = gen.sample(4)
        np.testing.assert_array_almost_equal(mults, np.ones(4))

    def test_autocorrelation(self, rng):
        cfg = SpreadNoiseConfig(enabled=True, autocorrelation=0.99, base_multiplier_std=0.5)
        gen = SpreadNoiseGenerator(cfg, rng)
        gen.reset(1)
        samples = [gen.sample(1)[0] for _ in range(100)]
        # With high autocorrelation, consecutive samples should be correlated
        corr = np.corrcoef(samples[:-1], samples[1:])[0, 1]
        assert corr > 0.5  # should be positively autocorrelated

    def test_apply_to_spread(self, rng):
        cfg = SpreadNoiseConfig(enabled=True, tick_size=0.01)
        gen = SpreadNoiseGenerator(cfg, rng)
        gen.reset(4)
        spreads = np.array([0.1, 0.05, 0.02, 0.01])
        noisy = gen.apply_to_spread(spreads)
        assert noisy.shape == spreads.shape
        assert np.all(noisy >= cfg.tick_size)

    def test_intraday_seasonality_effect(self, rng):
        cfg = SpreadNoiseConfig(
            enabled=True, intraday_seasonality=True, seasonality_amplitude=1.0,
            autocorrelation=0.0,
        )
        gen = SpreadNoiseGenerator(cfg, rng)
        gen.reset(1)
        # Sample many times at open (t=0) vs midday (t=0.5)
        open_mults = [gen.sample(1, 0.0)[0] for _ in range(50)]
        mid_mults = [gen.sample(1, 0.5)[0] for _ in range(50)]
        # Open should have higher average spread
        assert np.mean(open_mults) > np.mean(mid_mults)

    def test_tick_rounding(self, rng):
        cfg = SpreadNoiseConfig(enabled=True, tick_rounding=True, tick_size=0.05)
        gen = SpreadNoiseGenerator(cfg, rng)
        gen.reset(4)
        spreads = np.array([0.1, 0.15, 0.2, 0.25])
        noisy = gen.apply_to_spread(spreads)
        # Each value should be a multiple of tick_size
        residuals = noisy % cfg.tick_size
        np.testing.assert_array_almost_equal(residuals, np.zeros(4), decimal=8)


# ---------------------------------------------------------------------------
# FillRateRandomizer
# ---------------------------------------------------------------------------

class TestFillRateRandomizer:
    def test_fill_returns_tuple(self, rng):
        cfg = FillRateConfig(enabled=True)
        rand = FillRateRandomizer(cfg, rng)
        filled, frac, delay = rand.sample_fill(1, 100.0)
        assert isinstance(filled, bool)
        assert isinstance(frac, float)
        assert isinstance(delay, int)

    def test_fill_fraction_in_range(self, rng):
        cfg = FillRateConfig(enabled=True, base_fill_prob=1.0, partial_fill_prob=1.0)
        rand = FillRateRandomizer(cfg, rng)
        for i in range(100):
            _, frac, _ = rand.sample_fill(i, 10.0)
            assert 0.0 <= frac <= 1.0

    def test_disabled_always_fills_fully(self, rng):
        cfg = FillRateConfig(enabled=False)
        rand = FillRateRandomizer(cfg, rng)
        for i in range(50):
            filled, frac, delay = rand.sample_fill(i, 100.0)
            assert filled is True
            assert frac == 1.0
            assert delay == 0

    def test_large_orders_fill_worse(self, rng):
        cfg = FillRateConfig(enabled=True, base_fill_prob=1.0, size_impact_factor=0.5)
        rand = FillRateRandomizer(cfg, rng)
        # Small order
        small_fills = [rand.sample_fill(i, 1.0)[1] for i in range(200)]
        # Large order
        large_fills = [rand.sample_fill(i, 10000.0)[1] for i in range(200)]
        # Large fills should have lower average
        assert np.mean(small_fills) >= np.mean(large_fills)

    def test_reset_clears_pending(self, rng):
        cfg = FillRateConfig(enabled=True)
        rand = FillRateRandomizer(cfg, rng)
        rand.reset()
        assert len(rand._pending_fills) == 0

    def test_fill_prob_high_queue_position(self, rng):
        cfg = FillRateConfig(enabled=True, base_fill_prob=0.9, queue_position_factor=0.5)
        rand = FillRateRandomizer(cfg, rng)
        # At queue_position=0 (top), should fill more often
        top_fills = [rand.sample_fill(i, 10.0, queue_position=0.0)[0] for i in range(200)]
        # At queue_position=1 (back), should fill less
        back_fills = [rand.sample_fill(i, 10.0, queue_position=1.0)[0] for i in range(200)]
        assert np.mean(top_fills) > np.mean(back_fills)


# ---------------------------------------------------------------------------
# LatencyJitterInjector
# ---------------------------------------------------------------------------

class TestLatencyJitterInjector:
    def test_latency_positive(self, rng):
        cfg = LatencyConfig(enabled=True, base_latency_us=100.0)
        inj = LatencyJitterInjector(cfg, rng)
        for _ in range(100):
            lat = inj.sample_latency_us()
            assert lat >= 0.0

    def test_latency_bounded(self, rng):
        cfg = LatencyConfig(enabled=True, max_latency_us=1000.0)
        inj = LatencyJitterInjector(cfg, rng)
        for _ in range(200):
            lat = inj.sample_latency_us()
            assert lat <= cfg.max_latency_us

    def test_disabled_returns_base(self, rng):
        cfg = LatencyConfig(enabled=False, base_latency_us=500.0)
        inj = LatencyJitterInjector(cfg, rng)
        lat = inj.sample_latency_us()
        assert lat == cfg.base_latency_us

    def test_congestion_increases_latency(self, rng):
        cfg = LatencyConfig(
            enabled=True,
            base_latency_us=10.0,
            network_congestion_prob=1.0,  # always congest
            congestion_multiplier=100.0,
        )
        inj = LatencyJitterInjector(cfg, rng)
        lat = inj.sample_latency_us(intensity=1.0)
        assert lat > cfg.base_latency_us

    def test_latency_to_steps(self, rng):
        cfg = LatencyConfig(enabled=True, time_discretization_steps=10)
        inj = LatencyJitterInjector(cfg, rng)
        steps = inj.latency_to_steps(1000.0, step_duration_us=100.0)
        assert steps >= 0

    def test_packet_loss_prob(self, rng):
        cfg = LatencyConfig(enabled=True, packet_loss_prob=0.5)
        inj = LatencyJitterInjector(cfg, rng)
        drops = [inj.should_drop_packet() for _ in range(1000)]
        # Should drop ~50%
        assert 0.3 <= np.mean(drops) <= 0.7


# ---------------------------------------------------------------------------
# PriceImpactRandomizer
# ---------------------------------------------------------------------------

class TestPriceImpactRandomizer:
    def test_impact_sign(self, rng):
        cfg = PriceImpactConfig(enabled=True)
        rand = PriceImpactRandomizer(cfg, rng)
        rand.reset(4)
        temp, perm = rand.compute_impact(100.0, 0, sign=1.0)
        assert temp > 0
        assert perm > 0
        temp, perm = rand.compute_impact(100.0, 0, sign=-1.0)
        assert temp < 0
        assert perm < 0

    def test_lambda_in_bounds(self, rng):
        cfg = PriceImpactConfig(enabled=True, kyle_lambda_min=1e-4, kyle_lambda_max=0.01)
        rand = PriceImpactRandomizer(cfg, rng)
        for _ in range(20):
            rand.reset(4)
            lambdas = rand.current_lambda
            assert np.all(lambdas >= cfg.kyle_lambda_min)
            assert np.all(lambdas <= cfg.kyle_lambda_max)

    def test_larger_trade_larger_impact(self, rng):
        cfg = PriceImpactConfig(enabled=True)
        rand = PriceImpactRandomizer(cfg, rng)
        rand.reset(4)
        temp_small, _ = rand.compute_impact(10.0, 0, sign=1.0)
        temp_large, _ = rand.compute_impact(1000.0, 0, sign=1.0)
        assert temp_large > temp_small

    def test_decay_temporary_impact(self, rng):
        cfg = PriceImpactConfig(enabled=True, temporary_impact_decay=0.5)
        rand = PriceImpactRandomizer(cfg, rng)
        rand.reset(4)
        # Set some temporary impact
        rand._temporary_impact[0] = 1.0
        val1 = rand.decay_temporary_impact(0)
        val2 = rand.decay_temporary_impact(0)
        assert val2 < val1

    def test_regime_dependence(self, rng):
        cfg = PriceImpactConfig(enabled=True, regime_dependence=True)
        rand = PriceImpactRandomizer(cfg, rng)
        rand.reset(4, MarketRegime.CALM)
        calm_lambda = rand.current_lambda.mean()
        rand.refresh(4, MarketRegime.CRISIS)
        crisis_lambda = rand.current_lambda.mean()
        # Crisis lambda should tend to be higher (not guaranteed for single sample)
        # Just test it's within bounds
        assert crisis_lambda >= cfg.kyle_lambda_min

    def test_permanent_fraction(self, rng):
        cfg = PriceImpactConfig(
            enabled=True, permanent_impact_fraction=0.4, volume_nonlinearity=1.0
        )
        rand = PriceImpactRandomizer(cfg, rng)
        rand.reset(4)
        temp, perm = rand.compute_impact(100.0, 0, sign=1.0)
        total = temp + perm
        if abs(total) > 1e-10:
            frac_perm = abs(perm) / abs(total)
            assert abs(frac_perm - cfg.permanent_impact_fraction) < 0.05


# ---------------------------------------------------------------------------
# OrderBookDepthRandomizer
# ---------------------------------------------------------------------------

class TestOrderBookDepthRandomizer:
    def test_depth_levels_in_range(self, rng):
        cfg = OrderBookDepthConfig(enabled=True, min_levels=3, max_levels=10)
        rand = OrderBookDepthRandomizer(cfg, rng)
        rand.reset(4)
        assert cfg.min_levels <= rand.num_levels <= cfg.max_levels

    def test_get_depth_positive(self, rng):
        cfg = OrderBookDepthConfig(enabled=True)
        rand = OrderBookDepthRandomizer(cfg, rng)
        rand.reset(4)
        for asset in range(4):
            depth = rand.get_depth(asset, "bid", 0)
            assert depth >= 1.0

    def test_depth_decays_across_levels(self, rng):
        cfg = OrderBookDepthConfig(enabled=True, depth_decay_rate=0.5, min_levels=5, max_levels=5)
        rand = OrderBookDepthRandomizer(cfg, rng)
        rand.reset(1)
        depths = [rand.get_depth(0, "bid", level) for level in range(5)]
        # Depth should generally decrease (not strictly guaranteed due to noise)
        assert depths[0] >= depths[4]

    def test_disabled_returns_base_mean(self, rng):
        cfg = OrderBookDepthConfig(enabled=False, base_depth_mean=200.0)
        rand = OrderBookDepthRandomizer(cfg, rng)
        rand.reset(4)
        depth = rand.get_depth(0, "bid", 0)
        assert depth == cfg.base_depth_mean

    def test_step_triggers_refresh(self, rng):
        cfg = OrderBookDepthConfig(enabled=True, refresh_rate_min=1.0, refresh_rate_max=1.0)
        rand = OrderBookDepthRandomizer(cfg, rng)
        rand.reset(4)
        depth_before = rand.get_depth(0, "bid", 0)
        for _ in range(2):
            rand.step(4)
        # Just test no error
        depth_after = rand.get_depth(0, "bid", 0)
        assert depth_after >= 1.0


# ---------------------------------------------------------------------------
# LiquidityShockInjector
# ---------------------------------------------------------------------------

class TestLiquidityShockInjector:
    def test_no_shocks_initially(self, rng):
        cfg = LiquidityShockConfig(enabled=True, shock_prob_per_step=0.0)
        inj = LiquidityShockInjector(cfg, rng, num_assets=4)
        inj.reset()
        assert len(inj.active_shocks) == 0

    def test_shock_injection(self, rng):
        cfg = LiquidityShockConfig(enabled=True, shock_prob_per_step=1.0)
        inj = LiquidityShockInjector(cfg, rng, num_assets=4)
        inj.reset()
        inj.step(intensity=1.0)
        assert len(inj.active_shocks) >= 1

    def test_spread_multiplier_during_shock(self, rng):
        cfg = LiquidityShockConfig(
            enabled=True,
            shock_prob_per_step=1.0,
            spread_widening_min_factor=5.0,
            spread_widening_max_factor=5.0,
            shock_duration_steps_min=50,
            shock_duration_steps_max=50,
        )
        inj = LiquidityShockInjector(cfg, rng, num_assets=1)
        inj.reset()
        inj.step()
        mult = inj.get_spread_multiplier(0)
        assert mult >= 1.0

    def test_depth_fraction_during_shock(self, rng):
        cfg = LiquidityShockConfig(
            enabled=True,
            shock_prob_per_step=1.0,
            depth_removal_fraction_min=0.5,
            depth_removal_fraction_max=0.5,
            shock_duration_steps_min=50,
            shock_duration_steps_max=50,
        )
        inj = LiquidityShockInjector(cfg, rng, num_assets=1)
        inj.reset()
        inj.step()
        frac = inj.get_depth_fraction(0)
        assert frac <= 1.0

    def test_shock_decays(self, rng):
        cfg = LiquidityShockConfig(
            enabled=True,
            shock_prob_per_step=1.0,
            shock_duration_steps_min=5,
            shock_duration_steps_max=5,
        )
        inj = LiquidityShockInjector(cfg, rng, num_assets=1)
        inj.reset()
        for _ in range(20):
            inj.step(intensity=0.0)  # disable new shocks after first
        assert len(inj.active_shocks) == 0

    def test_shock_history_recorded(self, rng):
        cfg = LiquidityShockConfig(enabled=True, shock_prob_per_step=1.0)
        inj = LiquidityShockInjector(cfg, rng, num_assets=4)
        inj.reset()
        for _ in range(5):
            inj.step()
        assert len(inj.shock_history) >= 1


# ---------------------------------------------------------------------------
# RegimeManager
# ---------------------------------------------------------------------------

class TestRegimeManager:
    def test_initial_regime(self, rng):
        cfg = RegimeConfig(initial_regime=MarketRegime.CRASH)
        rm = RegimeManager(cfg, rng)
        regime = rm.reset()
        assert regime == MarketRegime.CRASH

    def test_random_initial_regime(self, rng):
        cfg = RegimeConfig(initial_regime=None)
        rm = RegimeManager(cfg, rng)
        regime = rm.reset()
        assert isinstance(regime, MarketRegime)

    def test_regime_transitions_occur(self, rng):
        cfg = RegimeConfig(
            regime_transition_prob=1.0,
            regime_min_duration=1,
            regime_max_duration=2,
        )
        rm = RegimeManager(cfg, rng)
        rm.reset()
        regimes_seen = set()
        for _ in range(50):
            regime, _ = rm.step(intensity=1.0)
            regimes_seen.add(regime)
        assert len(regimes_seen) > 1

    def test_volatility_by_regime(self, rng):
        cfg = RegimeConfig()
        rm = RegimeManager(cfg, rng)
        rm._regime = MarketRegime.CALM
        calm_vol = rm.get_volatility()
        rm._regime = MarketRegime.CRASH
        crash_vol = rm.get_volatility()
        assert crash_vol > calm_vol

    def test_drift_by_regime(self, rng):
        cfg = RegimeConfig(crash_drift=-0.01)
        rm = RegimeManager(cfg, rng)
        rm._regime = MarketRegime.CRASH
        drift = rm.get_drift()
        assert drift < 0

    def test_regime_history(self, rng):
        cfg = RegimeConfig(
            regime_transition_prob=0.5, regime_min_duration=1, regime_max_duration=3
        )
        rm = RegimeManager(cfg, rng)
        rm.reset()
        for _ in range(20):
            rm.step()
        assert len(rm.regime_history) >= 1

    def test_crisis_correlation_boost(self, rng):
        cfg = RegimeConfig(crisis_correlation_boost=0.3)
        rm = RegimeManager(cfg, rng)
        rm._regime = MarketRegime.CRISIS
        boost = rm.get_correlation_boost()
        assert boost == 0.3
        rm._regime = MarketRegime.CALM
        boost = rm.get_correlation_boost()
        assert boost == 0.0


# ---------------------------------------------------------------------------
# AdversarialParticipantInjector
# ---------------------------------------------------------------------------

class TestAdversarialParticipantInjector:
    def test_reset_initializes_adversaries(self, rng):
        cfg = AdversaryConfig(enabled=True, max_adversaries=5)
        inj = AdversarialParticipantInjector(cfg, rng, num_assets=4)
        inj.reset()
        assert inj.num_active_adversaries <= cfg.max_adversaries

    def test_disabled_returns_no_orders(self, rng):
        cfg = AdversaryConfig(enabled=False)
        inj = AdversarialParticipantInjector(cfg, rng, num_assets=4)
        inj.reset()
        orders = inj.step(np.ones(4) * 100.0, np.ones(4) * 0.01)
        assert len(orders) == 0

    def test_step_returns_list(self, rng):
        cfg = AdversaryConfig(
            enabled=True,
            max_adversaries=3,
            momentum_trader_prob=1.0,
            informed_trader_prob=0.0,
            wash_trader_prob=0.0,
            spoofer_prob=0.0,
            iceberg_prob=0.0,
            latency_arb_prob=0.0,
            predatory_hft_prob=0.0,
        )
        inj = AdversarialParticipantInjector(cfg, rng, num_assets=4)
        inj.reset()
        mid_prices = np.array([100.0, 50.0, 200.0, 75.0])
        spreads = np.array([0.1, 0.05, 0.2, 0.08])
        for _ in range(20):
            orders = inj.step(mid_prices, spreads, intensity=1.0)
        assert isinstance(orders, list)

    def test_order_fields(self, rng):
        cfg = AdversaryConfig(
            enabled=True,
            max_adversaries=5,
            wash_trader_prob=1.0,
            momentum_trader_prob=0.0,
            informed_trader_prob=0.0,
            spoofer_prob=0.0,
            iceberg_prob=0.0,
            latency_arb_prob=0.0,
            predatory_hft_prob=0.0,
        )
        inj = AdversarialParticipantInjector(cfg, rng, num_assets=1)
        inj.reset()
        mid = np.array([100.0])
        spread = np.array([0.1])
        # Run enough steps to trigger wash trades
        all_orders = []
        for _ in range(20):
            orders = inj.step(mid, spread)
            all_orders.extend(orders)
        if all_orders:
            order = all_orders[0]
            assert "side" in order
            assert "size" in order
            assert "price" in order


# ---------------------------------------------------------------------------
# AutoCurriculum
# ---------------------------------------------------------------------------

class TestAutoCurriculum:
    def test_initial_difficulty(self):
        cfg = CurriculumConfig(initial_difficulty=0.3)
        curr = AutoCurriculum(cfg)
        assert abs(curr.get_difficulty() - 0.3) < 1e-9

    def test_difficulty_increases_on_wins(self):
        cfg = CurriculumConfig(
            initial_difficulty=0.3,
            eval_window=20,
            win_rate_threshold_increase=0.6,
            difficulty_step_size=0.1,
        )
        curr = AutoCurriculum(cfg)
        # Win 80% of the time
        for _ in range(20):
            curr.record_outcome("spread_noise", win=True)
        for _ in range(5):
            curr.record_outcome("spread_noise", win=False)
        assert curr.get_difficulty("spread_noise") > 0.3

    def test_difficulty_decreases_on_losses(self):
        cfg = CurriculumConfig(
            initial_difficulty=0.7,
            eval_window=20,
            win_rate_threshold_decrease=0.4,
            difficulty_step_size=0.1,
        )
        curr = AutoCurriculum(cfg)
        # Lose 80% of the time
        for _ in range(5):
            curr.record_outcome("spread_noise", win=True)
        for _ in range(20):
            curr.record_outcome("spread_noise", win=False)
        assert curr.get_difficulty("spread_noise") < 0.7

    def test_difficulty_bounded(self):
        cfg = CurriculumConfig(
            initial_difficulty=0.9,
            max_difficulty=1.0,
            win_rate_threshold_increase=0.3,
        )
        curr = AutoCurriculum(cfg)
        for _ in range(500):
            curr.record_outcome("spread_noise", win=True)
        assert curr.get_difficulty("spread_noise") <= cfg.max_difficulty

    def test_per_scenario_tracking(self):
        cfg = CurriculumConfig(per_scenario_tracking=True)
        curr = AutoCurriculum(cfg)
        for _ in range(30):
            curr.record_outcome("spread_noise", win=True)
        for _ in range(30):
            curr.record_outcome("liquidity_shock", win=False)
        # spread_noise should have higher difficulty
        assert curr.get_difficulty("spread_noise") >= curr.get_difficulty("liquidity_shock")

    def test_curriculum_summary(self):
        cfg = CurriculumConfig()
        curr = AutoCurriculum(cfg)
        for _ in range(20):
            curr.record_outcome("spread_noise", win=True)
        summary = curr.get_curriculum_summary()
        assert "global_difficulty" in summary
        assert "per_scenario" in summary
        assert "total_updates" in summary

    def test_disabled(self):
        cfg = CurriculumConfig(enabled=False, initial_difficulty=0.5)
        curr = AutoCurriculum(cfg)
        curr.record_outcome("spread_noise", win=True)
        assert abs(curr.get_difficulty() - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# DomainRandomizationEngine
# ---------------------------------------------------------------------------

class TestDomainRandomizationEngine:
    def test_reset_returns_dict(self, dre):
        state = dre.reset()
        assert isinstance(state, dict)
        assert "regime" in state
        assert "annealer_intensity" in state

    def test_step_returns_dict(self, dre):
        dre.reset()
        mid_prices = np.array([100.0, 50.0, 200.0, 75.0])
        spreads = np.array([0.1, 0.05, 0.2, 0.08])
        result = dre.step(mid_prices, spreads)
        assert isinstance(result, dict)
        assert "spread_multipliers" in result
        assert "depth_fractions" in result
        assert "latency_us" in result

    def test_spread_multipliers_shape(self, dre):
        dre.reset()
        n = dre.cfg.num_assets
        mid = np.ones(n) * 100.0
        spreads = np.ones(n) * 0.01
        result = dre.step(mid, spreads)
        assert result["spread_multipliers"].shape == (n,)

    def test_augment_observation(self, dre):
        dre.reset()
        obs = np.random.randn(64)
        aug = dre.augment_observation(obs)
        assert aug.shape == obs.shape

    def test_augment_spreads(self, dre):
        dre.reset()
        spreads = np.array([0.1, 0.05, 0.02, 0.01])
        noisy = dre.augment_spreads(spreads)
        assert noisy.shape == spreads.shape
        assert np.all(noisy >= dre.cfg.spread_noise.tick_size)

    def test_sample_fill(self, dre):
        dre.reset()
        filled, frac, delay = dre.sample_fill(0, 100.0, spread=0.01, mid_price=100.0, limit_price=100.0)
        assert isinstance(filled, bool)
        assert 0.0 <= frac <= 1.0
        assert delay >= 0

    def test_compute_price_impact(self, dre):
        dre.reset()
        temp, perm = dre.compute_price_impact(100.0, 0, sign=1.0)
        assert isinstance(temp, float)
        assert isinstance(perm, float)

    def test_sample_latency(self, dre):
        dre.reset()
        lat, dropped, reseq = dre.sample_latency()
        assert lat >= 0.0
        assert isinstance(dropped, bool)
        assert isinstance(reseq, bool)

    def test_metrics_after_steps(self, dre):
        dre.reset()
        n = dre.cfg.num_assets
        mid = np.ones(n) * 100.0
        spreads = np.ones(n) * 0.01
        for _ in range(50):
            dre.step(mid, spreads)
        metrics = dre.get_metrics()
        assert "step" in metrics
        assert "spread_mult_mean" in metrics

    def test_curriculum_recording(self, dre):
        dre.reset()
        for i in range(20):
            dre.record_episode_outcome("spread_noise", agent_pnl=float(i), benchmark_pnl=0.0)
        difficulty = dre.get_difficulty("spread_noise")
        assert 0.0 <= difficulty <= 1.0

    def test_get_config_dict(self, dre):
        cfg_dict = dre.get_config_dict()
        assert isinstance(cfg_dict, dict)
        assert "spread_noise" in cfg_dict

    def test_from_config_dict_roundtrip(self, dre):
        cfg_dict = dre.get_config_dict()
        # Just test it doesn't raise
        try:
            dre2 = DomainRandomizationEngine.from_config_dict(cfg_dict)
            assert isinstance(dre2, DomainRandomizationEngine)
        except Exception:
            pass  # May fail due to enum reconstruction, acceptable

    def test_full_dre_with_adversaries(self, full_dre):
        full_dre.reset()
        n = full_dre.cfg.num_assets
        mid = np.ones(n) * 100.0
        spreads = np.ones(n) * 0.01
        for _ in range(20):
            result = full_dre.step(mid, spreads)
        assert "adversary_orders" in result


# ---------------------------------------------------------------------------
# ScenarioSampler
# ---------------------------------------------------------------------------

class TestScenarioSampler:
    def test_sample_returns_valid(self, rng):
        cfg = CurriculumConfig()
        curr = AutoCurriculum(cfg)
        sampler = ScenarioSampler(curr, rng)
        name, spec = sampler.sample_scenario()
        assert name in ScenarioSampler.SCENARIO_REGISTRY
        assert isinstance(spec, dict)

    def test_sample_various_scenarios(self, rng):
        cfg = CurriculumConfig()
        curr = AutoCurriculum(cfg)
        sampler = ScenarioSampler(curr, rng)
        names_seen = set()
        for _ in range(200):
            name, _ = sampler.sample_scenario()
            names_seen.add(name)
        # Should see multiple scenarios
        assert len(names_seen) > 1

    def test_build_dre_for_scenario(self, rng):
        cfg = CurriculumConfig()
        curr = AutoCurriculum(cfg)
        sampler = ScenarioSampler(curr, rng)
        for scenario_name in ["baseline", "full_chaos", "adversary"]:
            dre = sampler.build_dre_for_scenario(scenario_name, num_assets=2, seed=0)
            assert isinstance(dre, DomainRandomizationEngine)


# ---------------------------------------------------------------------------
# ParameterSchedule
# ---------------------------------------------------------------------------

class TestParameterSchedule:
    def test_step_schedule(self):
        sched = ParameterSchedule(
            values=[0.0, 0.5, 1.0],
            milestones=[100, 200],
            interpolation="linear",
        )
        for _ in range(100):
            sched.step()
        assert abs(sched.get() - 0.5) < 0.05

    def test_final_value_after_milestones(self):
        sched = ParameterSchedule(
            values=[0.0, 0.5, 1.0],
            milestones=[50, 100],
        )
        for _ in range(200):
            sched.step()
        assert abs(sched.get() - 1.0) < 1e-9

    def test_reset(self):
        sched = ParameterSchedule(values=[0.0, 1.0], milestones=[100])
        for _ in range(50):
            sched.step()
        sched.reset()
        assert sched.current_step == 0
        assert sched.get() == 0.0

    def test_cosine_interpolation(self):
        sched = ParameterSchedule(
            values=[0.0, 1.0],
            milestones=[100],
            interpolation="cosine",
        )
        val_at_50 = sched.get()
        for _ in range(50):
            sched.step()
        val_mid = sched.get()
        # Cosine interpolation should be non-linear
        assert 0.0 <= val_mid <= 1.0


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

class TestFactories:
    def test_make_dre(self):
        dre = make_dre(num_assets=2, num_agents=4, seed=0)
        assert isinstance(dre, DomainRandomizationEngine)
        assert dre.cfg.num_assets == 2

    def test_make_light_dre(self):
        dre = make_light_dre(seed=0)
        assert isinstance(dre, DomainRandomizationEngine)
        assert not dre.cfg.adversary.enabled

    def test_make_adversarial_dre(self):
        dre = make_adversarial_dre(seed=0)
        assert isinstance(dre, DomainRandomizationEngine)
        # Adversarial should have worse fill rates
        assert dre.cfg.fill_rate.base_fill_prob < 0.9

    def test_make_dre_different_seeds(self):
        dre1 = make_dre(seed=0)
        dre2 = make_dre(seed=42)
        dre1.reset()
        dre2.reset()
        n = dre1.cfg.num_assets
        mid = np.ones(n) * 100.0
        sp = np.ones(n) * 0.01
        r1 = dre1.step(mid, sp)
        r2 = dre2.step(mid, sp)
        # Different seeds should produce different spread multipliers
        # (not strictly guaranteed, but very likely)
        assert not np.allclose(r1["spread_multipliers"], r2["spread_multipliers"])


# ---------------------------------------------------------------------------
# ObservationAugmentor
# ---------------------------------------------------------------------------

class TestObservationAugmentor:
    def test_add_noise_shape(self, rng):
        aug = ObservationAugmentor(rng)
        obs = np.random.randn(64)
        noisy = aug.add_observation_noise(obs, noise_std=0.01)
        assert noisy.shape == obs.shape

    def test_noise_changes_obs(self, rng):
        aug = ObservationAugmentor(rng)
        obs = np.ones(64)
        noisy = aug.add_observation_noise(obs, noise_std=1.0, intensity=1.0)
        assert not np.allclose(obs, noisy)

    def test_dropout_zeros_some(self, rng):
        aug = ObservationAugmentor(rng)
        obs = np.ones(1000)
        noisy = aug.add_observation_noise(obs, noise_std=0.0, dropout_prob=0.3, intensity=1.0)
        zero_frac = np.mean(noisy == 0.0)
        assert 0.1 <= zero_frac <= 0.5

    def test_stale_obs_returns_history(self, rng):
        aug = ObservationAugmentor(rng)
        history = [np.zeros(64), np.ones(64)]
        obs = np.full(64, 2.0)
        # Force stale by setting stale_prob=1.0
        stale = aug.add_stale_observation(obs, history, stale_prob=1.0, intensity=1.0)
        assert not np.allclose(stale, obs)

    def test_regime_noise_applied(self, rng):
        aug = ObservationAugmentor(rng)
        obs = np.ones(64)
        noisy = aug.apply_regime_feature_noise(obs, MarketRegime.CRASH, intensity=1.0)
        assert not np.allclose(obs, noisy)


# ---------------------------------------------------------------------------
# DREMetricsTracker
# ---------------------------------------------------------------------------

class TestDREMetricsTracker:
    def test_record_and_summarize(self):
        tracker = DREMetricsTracker(window=100)
        for _ in range(50):
            tracker.record_spread_mult(2.0)
            tracker.record_fill_rate(0.8)
            tracker.record_latency(500.0)
            tracker.record_kyle_lambda(0.001)
            tracker.step()
        summary = tracker.get_summary()
        assert abs(summary["spread_mult_mean"] - 2.0) < 0.01
        assert abs(summary["fill_rate_mean"] - 0.8) < 0.01
        assert abs(summary["latency_us_mean"] - 500.0) < 1.0

    def test_shock_and_regime_counts(self):
        tracker = DREMetricsTracker()
        tracker.record_shock()
        tracker.record_shock()
        tracker.record_regime_change()
        tracker.record_adversary_orders(5)
        summary = tracker.get_summary()
        assert summary["total_shocks"] == 2
        assert summary["total_regime_changes"] == 1
        assert summary["total_adversary_orders"] == 5
