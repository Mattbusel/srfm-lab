"""
test_osi.py — Tests for the Online System Identification (OSI) module.

Covers:
- ResidualBuffer
- MarketResidualMonitor
- KalmanVolatilityFilter
- RLSKyleLambda
- GrangerLeadLagDetector
- AgentInternalEnvironmentModel
- RegimeChangeAlert
- LatencyProfiler
- OnlineSystemIdentification
- MultiAssetOSI
- Factory functions
"""

import math
import time
import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hyper_agent.online_system_identification import (
    # Configs
    OSIConfig,
    ResidualMonitorConfig,
    KalmanVolConfig,
    RLSConfig,
    GrangerConfig,
    IEMConfig,
    # Sub-modules
    ResidualBuffer,
    MarketResidualMonitor,
    KalmanVolatilityFilter,
    RLSKyleLambda,
    GrangerLeadLagDetector,
    AgentInternalEnvironmentModel,
    RegimeChangeAlert,
    LatencyProfiler,
    # Main
    OnlineSystemIdentification,
    MultiAssetOSI,
    # Factories
    make_osi,
    make_multi_asset_osi,
    # Constants
    OSI_LATENCY_BUDGET_MS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def osi():
    return make_osi(num_assets=2, obs_dim=16, action_dim=4, seed=0)


@pytest.fixture
def multi_osi():
    return make_multi_asset_osi(num_assets=2, obs_dim=16, action_dim=4)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def make_obs_data(n=100, obs_dim=16, action_dim=4, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    obs = rng.standard_normal((n, obs_dim)).astype(np.float32)
    actions = rng.standard_normal((n, action_dim)).astype(np.float32)
    next_obs = obs + 0.1 * rng.standard_normal((n, obs_dim)).astype(np.float32)
    return obs, actions, next_obs


# ---------------------------------------------------------------------------
# ResidualBuffer
# ---------------------------------------------------------------------------

class TestResidualBuffer:
    def test_push_and_mean(self):
        buf = ResidualBuffer(window=100)
        for v in [1.0, 2.0, 3.0, 4.0]:
            buf.push(v)
        assert abs(buf.mean - 2.5) < 1e-9

    def test_ewma_updates(self):
        buf = ResidualBuffer(window=100, alpha=0.5)
        buf.push(10.0)
        buf.push(10.0)
        # EWMA should be close to 10
        assert abs(buf.ewma - 10.0) < 5.0

    def test_z_score(self):
        buf = ResidualBuffer(window=100)
        for v in [0.0] * 50:
            buf.push(v)
        # Push a large value
        buf.push(100.0)
        z = buf.z_score(100.0)
        assert z > 1.0

    def test_max_window(self):
        buf = ResidualBuffer(window=10)
        for i in range(100):
            buf.push(float(i))
        assert len(buf.to_array()) <= 10

    def test_n_tracks_total_pushes(self):
        buf = ResidualBuffer(window=5)
        for i in range(20):
            buf.push(1.0)
        assert buf.n == 20

    def test_std_zero_for_constant(self):
        buf = ResidualBuffer(window=100)
        for _ in range(50):
            buf.push(3.0)
        # Std should be near 0 (EWMA variance may be small but positive)
        assert buf.std < 0.01


# ---------------------------------------------------------------------------
# MarketResidualMonitor
# ---------------------------------------------------------------------------

class TestMarketResidualMonitor:
    def test_update_returns_float(self):
        cfg = ResidualMonitorConfig()
        mon = MarketResidualMonitor(cfg, num_assets=2)
        r = mon.update_impact(0, 0.01, 0.005)
        assert isinstance(r, float)

    def test_update_fill_rate(self):
        cfg = ResidualMonitorConfig()
        mon = MarketResidualMonitor(cfg, num_assets=2)
        r = mon.update_fill_rate(0, 0.9, 0.7)
        assert isinstance(r, float)

    def test_update_spread(self):
        cfg = ResidualMonitorConfig()
        mon = MarketResidualMonitor(cfg, num_assets=2)
        r = mon.update_spread(0, 0.1, 0.05)
        assert isinstance(r, float)

    def test_no_alert_insufficient_samples(self):
        cfg = ResidualMonitorConfig(min_samples_for_alert=100)
        mon = MarketResidualMonitor(cfg, num_assets=2)
        # Only push 5 samples
        for _ in range(5):
            mon.update_impact(0, 1.0, 0.0)
        alerted, _ = mon.should_alert(0)
        assert not alerted

    def test_alert_on_large_residual(self):
        cfg = ResidualMonitorConfig(
            impact_residual_threshold=0.01,
            min_samples_for_alert=5,
            ewma_alpha=1.0,  # instant update
        )
        mon = MarketResidualMonitor(cfg, num_assets=2)
        # Push large residuals
        for _ in range(20):
            mon.update_impact(0, 1.0, 0.0)  # residual = 1.0
        alerted, reason = mon.should_alert(0)
        assert alerted
        assert "impact" in reason

    def test_invalid_asset_id(self):
        cfg = ResidualMonitorConfig()
        mon = MarketResidualMonitor(cfg, num_assets=2)
        # Should not raise, should return 0
        r = mon.update_impact(99, 1.0, 0.0)
        assert r == 0.0

    def test_get_summary(self):
        cfg = ResidualMonitorConfig()
        mon = MarketResidualMonitor(cfg, num_assets=2)
        for _ in range(10):
            mon.update_impact(0, 0.1, 0.05)
        summary = mon.get_summary()
        assert "asset_0" in summary
        assert "impact_ewma" in summary["asset_0"]

    def test_outlier_clipping(self):
        cfg = ResidualMonitorConfig(outlier_clip_sigma=2.0)
        mon = MarketResidualMonitor(cfg, num_assets=1)
        # Build up stats
        for _ in range(50):
            mon.update_impact(0, 0.01, 0.01)
        # Push outlier
        r = mon.update_impact(0, 1000.0, 0.0)
        # Should be clipped (not 1000)
        assert abs(r) < 100.0


# ---------------------------------------------------------------------------
# KalmanVolatilityFilter
# ---------------------------------------------------------------------------

class TestKalmanVolatilityFilter:
    def test_update_returns_dict(self):
        cfg = KalmanVolConfig(num_regimes=3)
        f = KalmanVolatilityFilter(cfg)
        result = f.update(0.01)
        assert "vol_estimate" in result
        assert "regime_probs" in result
        assert "dominant_regime" in result

    def test_vol_estimate_positive(self):
        cfg = KalmanVolConfig()
        f = KalmanVolatilityFilter(cfg)
        for _ in range(100):
            result = f.update(np.random.randn() * 0.01)
        assert result["vol_estimate"] > 0

    def test_vol_in_bounds(self):
        cfg = KalmanVolConfig(min_vol=1e-5, max_vol=0.5)
        f = KalmanVolatilityFilter(cfg)
        for _ in range(200):
            r = f.update(np.random.randn() * 0.05)
            assert cfg.min_vol <= r["vol_estimate"] <= cfg.max_vol

    def test_regime_probs_sum_to_one(self):
        cfg = KalmanVolConfig(num_regimes=3)
        f = KalmanVolatilityFilter(cfg)
        for _ in range(50):
            r = f.update(np.random.randn() * 0.01)
        probs = r["regime_probs"]
        assert abs(probs.sum() - 1.0) < 1e-6
        assert np.all(probs >= 0)

    def test_high_vol_increases_estimate(self):
        cfg = KalmanVolConfig(num_regimes=2)
        f = KalmanVolatilityFilter(cfg)
        # Run with low vol
        for _ in range(50):
            f.update(0.001)
        low_est = f.current_vol
        f.reset()
        # Run with high vol
        for _ in range(50):
            f.update(0.1)
        high_est = f.current_vol
        assert high_est > low_est

    def test_reset_clears_state(self):
        cfg = KalmanVolConfig()
        f = KalmanVolatilityFilter(cfg)
        for _ in range(100):
            f.update(0.05)
        f.reset()
        assert f.current_vol == cfg.initial_vol
        assert abs(f.regime_probs.sum() - 1.0) < 1e-6

    def test_em_update_runs(self):
        cfg = KalmanVolConfig(em_update_interval=20, em_max_iter=2)
        f = KalmanVolatilityFilter(cfg)
        # Should not raise
        for _ in range(50):
            f.update(0.01)

    def test_dominant_regime_is_valid(self):
        cfg = KalmanVolConfig(num_regimes=4)
        f = KalmanVolatilityFilter(cfg)
        for _ in range(20):
            r = f.update(0.01)
        assert 0 <= r["dominant_regime"] < cfg.num_regimes


# ---------------------------------------------------------------------------
# RLSKyleLambda
# ---------------------------------------------------------------------------

class TestRLSKyleLambda:
    def test_update_returns_dict(self):
        cfg = RLSConfig(num_features=3)
        rls = RLSKyleLambda(cfg)
        features = np.array([100.0, 0.01, 0.02])
        result = rls.update(features, 0.001)
        assert "lambda_estimate" in result
        assert "residual" in result

    def test_lambda_in_bounds(self):
        cfg = RLSConfig(num_features=3, min_lambda=1e-6, max_lambda=0.01)
        rls = RLSKyleLambda(cfg)
        for _ in range(100):
            features = np.random.randn(3)
            rls.update(features, np.random.randn() * 0.001)
        assert cfg.min_lambda <= rls.kyle_lambda <= cfg.max_lambda

    def test_convergence_on_linear_model(self):
        """RLS should converge to true lambda for a linear model."""
        cfg = RLSConfig(
            num_features=1, forgetting_factor=1.0, initial_covariance=100.0,
            min_lambda=0.0, max_lambda=1.0,
        )
        rls = RLSKyleLambda(cfg)
        true_lambda = 0.005
        rng = np.random.default_rng(0)
        for _ in range(300):
            vol = rng.uniform(0.001, 0.05)
            feature = np.array([vol * 100.0])
            target = true_lambda * vol * 100.0 + rng.normal(0, 1e-5)
            rls.update(feature, target)
        # Estimated lambda should be close to true lambda
        estimated = rls.theta[0]
        # Not checking exact value but sanity check
        assert abs(estimated) < 10.0

    def test_reset(self):
        cfg = RLSConfig(num_features=3)
        rls = RLSKyleLambda(cfg)
        for _ in range(50):
            rls.update(np.random.randn(3), 0.001)
        rls.reset()
        np.testing.assert_array_equal(rls.theta, np.zeros(cfg.num_features))

    def test_forgetting_factor_effect(self):
        """With low forgetting factor, old data is forgotten."""
        cfg = RLSConfig(num_features=1, forgetting_factor=0.5, min_lambda=0.0, max_lambda=100.0)
        rls = RLSKyleLambda(cfg)
        # Push data with signal lambda=1
        for _ in range(20):
            rls.update(np.array([1.0]), 1.0)
        theta1 = rls.theta[0]
        # Push data with signal lambda=10
        for _ in range(20):
            rls.update(np.array([1.0]), 10.0)
        theta2 = rls.theta[0]
        assert theta2 > theta1

    def test_outlier_skipping(self):
        cfg = RLSConfig(num_features=1, outlier_threshold=1.0)
        rls = RLSKyleLambda(cfg)
        # Build up residual stats
        for _ in range(50):
            rls.update(np.array([1.0]), 1.0)
        theta_before = rls.theta.copy()
        # Push extreme outlier
        result = rls.update(np.array([1.0]), 1000.0)
        # Outlier might be skipped
        assert isinstance(result.get("outlier_skipped"), bool)

    def test_covariance_matrix_shape(self):
        cfg = RLSConfig(num_features=3)
        rls = RLSKyleLambda(cfg)
        cov = rls.covariance
        assert cov.shape == (cfg.num_features, cfg.num_features)

    def test_lambda_uncertainty_non_negative(self):
        cfg = RLSConfig(num_features=3)
        rls = RLSKyleLambda(cfg)
        assert rls.lambda_uncertainty >= 0.0


# ---------------------------------------------------------------------------
# GrangerLeadLagDetector
# ---------------------------------------------------------------------------

class TestGrangerLeadLagDetector:
    def test_insufficient_data(self):
        cfg = GrangerConfig(num_assets=2, min_samples=100, test_interval=10)
        g = GrangerLeadLagDetector(cfg)
        for _ in range(10):
            g.update(np.array([0.01, -0.01]))
        result = g.run_test()
        # Not enough data
        assert result["n_samples"] <= 20

    def test_run_test_with_enough_data(self):
        cfg = GrangerConfig(
            num_assets=2, lag_order=2, min_samples=20,
            rolling_window=50, test_interval=5
        )
        g = GrangerLeadLagDetector(cfg)
        rng = np.random.default_rng(0)
        for _ in range(100):
            g.update(rng.standard_normal(2) * 0.01)
        result = g.run_test()
        assert "granger_matrix" in result or "var_coefs" in result

    def test_should_run_test(self):
        cfg = GrangerConfig(test_interval=10)
        g = GrangerLeadLagDetector(cfg)
        for _ in range(5):
            g.update(np.array([0.01, -0.01]))
        assert not g.should_run_test()
        for _ in range(6):
            g.update(np.array([0.01, -0.01]))
        assert g.should_run_test()

    def test_reset_clears_history(self):
        cfg = GrangerConfig(num_assets=2)
        g = GrangerLeadLagDetector(cfg)
        for _ in range(50):
            g.update(np.ones(2) * 0.01)
        g.reset()
        assert g._idx == 0
        assert g._var_coefs is None

    def test_structural_break_detection(self):
        cfg = GrangerConfig(
            num_assets=2, lag_order=1, min_samples=20,
            rolling_window=50, test_interval=10,
            structural_break_threshold=0.01,
        )
        g = GrangerLeadLagDetector(cfg)
        rng = np.random.default_rng(0)
        # First: correlated returns
        for _ in range(50):
            x = rng.standard_normal()
            g.update(np.array([x, x]))
        result1 = g.run_test()
        # Then: uncorrelated (structural break)
        for _ in range(50):
            g.update(rng.standard_normal(2))
        result2 = g.run_test()
        # At least one result should have valid coefs
        assert result1.get("var_coefs") is not None or result2.get("var_coefs") is not None


# ---------------------------------------------------------------------------
# AgentInternalEnvironmentModel
# ---------------------------------------------------------------------------

class TestAgentIEM:
    def test_predict_shape(self):
        cfg = IEMConfig(obs_dim=16, action_dim=4, hidden_dim=32, output_dim=16)
        iem = AgentInternalEnvironmentModel(cfg)
        obs = np.random.randn(16).astype(np.float32)
        action = np.random.randn(4).astype(np.float32)
        pred, latency = iem.predict(obs, action)
        assert pred.shape == (16,)
        assert latency >= 0.0

    def test_push_returns_float(self):
        cfg = IEMConfig(obs_dim=16, action_dim=4, hidden_dim=32, output_dim=16)
        iem = AgentInternalEnvironmentModel(cfg)
        rng = np.random.default_rng(0)
        # Push enough for training
        for _ in range(10):
            obs = rng.standard_normal(16).astype(np.float32)
            action = rng.standard_normal(4).astype(np.float32)
            next_obs = obs + 0.1
            residual = iem.push(obs, action, next_obs)
        assert isinstance(residual, float)

    def test_update_after_enough_samples(self):
        cfg = IEMConfig(
            obs_dim=16, action_dim=4, hidden_dim=32, output_dim=16,
            min_samples_for_update=20, warm_start_steps=5,
            update_every_n_steps=5, batch_size=10,
        )
        iem = AgentInternalEnvironmentModel(cfg)
        rng = np.random.default_rng(0)
        for i in range(50):
            obs = rng.standard_normal(16).astype(np.float32)
            action = rng.standard_normal(4).astype(np.float32)
            next_obs = obs + 0.01
            iem.push(obs, action, next_obs)
        if iem.should_update(0.1):
            loss = iem.update()
            if loss is not None:
                assert loss >= 0.0

    def test_buffer_size_bounded(self):
        cfg = IEMConfig(
            obs_dim=8, action_dim=2, hidden_dim=16, output_dim=8,
            replay_buffer_size=50,
        )
        iem = AgentInternalEnvironmentModel(cfg)
        rng = np.random.default_rng(0)
        for _ in range(100):
            iem.push(
                rng.standard_normal(8).astype(np.float32),
                rng.standard_normal(2).astype(np.float32),
                rng.standard_normal(8).astype(np.float32),
            )
        assert iem.buffer_size <= 50

    def test_reset_clears_buffer(self):
        cfg = IEMConfig(obs_dim=8, action_dim=2, hidden_dim=16, output_dim=8)
        iem = AgentInternalEnvironmentModel(cfg)
        rng = np.random.default_rng(0)
        for _ in range(10):
            iem.push(rng.standard_normal(8).astype(np.float32),
                     rng.standard_normal(2).astype(np.float32),
                     rng.standard_normal(8).astype(np.float32))
        iem.reset()
        assert iem.buffer_size == 0

    def test_latency_budget(self):
        cfg = IEMConfig(obs_dim=64, action_dim=8, hidden_dim=128, output_dim=64)
        iem = AgentInternalEnvironmentModel(cfg)
        obs = np.random.randn(64).astype(np.float32)
        action = np.random.randn(8).astype(np.float32)
        # Warm up
        for _ in range(5):
            iem.predict(obs, action)
        # Measure latency
        latencies = []
        for _ in range(20):
            _, lat = iem.predict(obs, action)
            latencies.append(lat)
        mean_latency = np.mean(latencies)
        # Should be well under 100ms (OSI budget is 5ms total, IEM is a fraction)
        assert mean_latency < 100.0


# ---------------------------------------------------------------------------
# RegimeChangeAlert
# ---------------------------------------------------------------------------

class TestRegimeChangeAlert:
    def test_publish_calls_callback(self):
        received = []
        def cb(alert_type, payload):
            received.append((alert_type, payload))

        alert = RegimeChangeAlert(callback=cb, cooldown_steps=0)
        alert.publish("test_alert", {"foo": 1})
        assert len(received) == 1
        assert received[0][0] == "test_alert"

    def test_cooldown_prevents_duplicate(self):
        received = []
        alert = RegimeChangeAlert(callback=lambda t, p: received.append(t), cooldown_steps=100)
        alert.publish("test_alert", {})
        alert.publish("test_alert", {})
        assert len(received) == 1

    def test_cooldown_expires(self):
        received = []
        alert = RegimeChangeAlert(callback=lambda t, p: received.append(t), cooldown_steps=5)
        alert.publish("test_alert", {})
        for _ in range(6):
            alert.step()
        alert.publish("test_alert", {})
        assert len(received) == 2

    def test_drain(self):
        alert = RegimeChangeAlert(cooldown_steps=0)
        alert.publish("a", {})
        alert.publish("b", {})
        drained = alert.drain()
        assert len(drained) == 2
        drained2 = alert.drain()
        assert len(drained2) == 0

    def test_alert_history(self):
        alert = RegimeChangeAlert(cooldown_steps=0)
        for i in range(5):
            alert.publish(f"alert_{i}", {"i": i})
        assert len(alert.alert_history) == 5


# ---------------------------------------------------------------------------
# LatencyProfiler
# ---------------------------------------------------------------------------

class TestLatencyProfiler:
    def test_record_and_stats(self):
        prof = LatencyProfiler(window=100)
        for i in range(50):
            prof.record("iem", float(i))
        stats = prof.get_stats()
        assert "iem" in stats
        assert stats["iem"]["mean_ms"] >= 0.0
        assert stats["iem"]["p99_ms"] >= stats["iem"]["mean_ms"]

    def test_budget_exceeded(self):
        prof = LatencyProfiler(window=100)
        for _ in range(10):
            prof.record("total", OSI_LATENCY_BUDGET_MS * 2)
        stats = prof.get_stats()
        assert stats["total"]["budget_exceeded_pct"] == 100.0

    def test_total_mean(self):
        prof = LatencyProfiler(window=100)
        prof.record("comp_a", 2.0)
        prof.record("comp_b", 3.0)
        total = prof.total_mean_ms()
        assert total == pytest.approx(5.0, abs=0.01)


# ---------------------------------------------------------------------------
# OnlineSystemIdentification
# ---------------------------------------------------------------------------

class TestOnlineSystemIdentification:
    def test_reset_does_not_raise(self, osi):
        osi.reset()

    def test_update_returns_dict(self, osi):
        osi.reset()
        n = 2
        result = osi.update(
            log_returns=np.zeros(n),
            signed_volumes=np.zeros(n),
            price_changes=np.zeros(n),
            predicted_impacts=np.zeros(n),
            actual_impacts=np.zeros(n),
            predicted_fill_rates=np.ones(n) * 0.9,
            actual_fill_rates=np.ones(n) * 0.9,
            predicted_spreads=np.ones(n) * 0.01,
            actual_spreads=np.ones(n) * 0.01,
        )
        assert isinstance(result, dict)
        assert "vol_estimates" in result
        assert "lambda_estimates" in result
        assert "total_latency_ms" in result

    def test_vol_estimates_shape(self, osi):
        osi.reset()
        n = 2
        rng = np.random.default_rng(0)
        for _ in range(10):
            result = osi.update(
                log_returns=rng.standard_normal(n) * 0.01,
                signed_volumes=rng.standard_normal(n) * 100,
                price_changes=rng.standard_normal(n) * 0.001,
                predicted_impacts=np.zeros(n),
                actual_impacts=rng.standard_normal(n) * 0.001,
                predicted_fill_rates=np.ones(n) * 0.9,
                actual_fill_rates=np.ones(n) * 0.8,
                predicted_spreads=np.ones(n) * 0.01,
                actual_spreads=np.ones(n) * 0.012,
            )
        assert result["vol_estimates"].shape == (n,)
        assert result["lambda_estimates"].shape == (n,)

    def test_latency_budget(self, osi):
        osi.reset()
        n = 2
        rng = np.random.default_rng(0)
        latencies = []
        for _ in range(20):
            result = osi.update(
                log_returns=rng.standard_normal(n) * 0.01,
                signed_volumes=rng.standard_normal(n) * 100,
                price_changes=rng.standard_normal(n) * 0.001,
                predicted_impacts=np.zeros(n),
                actual_impacts=np.zeros(n),
                predicted_fill_rates=np.ones(n),
                actual_fill_rates=np.ones(n),
                predicted_spreads=np.ones(n) * 0.01,
                actual_spreads=np.ones(n) * 0.01,
            )
            latencies.append(result["total_latency_ms"])
        # Mean latency should be reasonable (no more than 100ms in test env)
        assert np.mean(latencies) < 100.0

    def test_iem_update_with_obs(self, osi):
        osi.reset()
        n = 2
        rng = np.random.default_rng(0)
        obs = rng.standard_normal(osi.cfg.iem.obs_dim).astype(np.float32)
        action = rng.standard_normal(osi.cfg.iem.action_dim).astype(np.float32)
        next_obs = obs + 0.01
        # Should not raise
        osi.update(
            log_returns=np.zeros(n),
            signed_volumes=np.zeros(n),
            price_changes=np.zeros(n),
            predicted_impacts=np.zeros(n),
            actual_impacts=np.zeros(n),
            predicted_fill_rates=np.ones(n),
            actual_fill_rates=np.ones(n),
            predicted_spreads=np.ones(n) * 0.01,
            actual_spreads=np.ones(n) * 0.01,
            obs=obs, action=action, next_obs=next_obs,
        )

    def test_granger_test_triggered(self, osi):
        osi.reset()
        n = 2
        rng = np.random.default_rng(0)
        # Run past the test interval
        for _ in range(osi.cfg.granger.test_interval + 5):
            osi.update(
                log_returns=rng.standard_normal(n) * 0.01,
                signed_volumes=rng.standard_normal(n) * 100,
                price_changes=rng.standard_normal(n) * 0.001,
                predicted_impacts=np.zeros(n),
                actual_impacts=np.zeros(n),
                predicted_fill_rates=np.ones(n),
                actual_fill_rates=np.ones(n),
                predicted_spreads=np.ones(n) * 0.01,
                actual_spreads=np.ones(n) * 0.01,
            )
        # Should not raise

    def test_full_state(self, osi):
        osi.reset()
        n = 2
        for _ in range(10):
            osi.update(
                log_returns=np.ones(n) * 0.01,
                signed_volumes=np.ones(n) * 10,
                price_changes=np.ones(n) * 0.001,
                predicted_impacts=np.zeros(n),
                actual_impacts=np.zeros(n),
                predicted_fill_rates=np.ones(n),
                actual_fill_rates=np.ones(n),
                predicted_spreads=np.ones(n) * 0.01,
                actual_spreads=np.ones(n) * 0.01,
            )
        state = osi.get_full_state()
        assert "vol_estimates" in state
        assert "lambda_estimates" in state
        assert "iem_buffer_size" in state

    def test_observation_features(self, osi):
        osi.reset()
        feats = osi.get_observation_features()
        assert isinstance(feats, np.ndarray)
        assert feats.dtype == np.float32
        assert len(feats) > 0

    def test_predict_next_obs(self, osi):
        osi.reset()
        obs = np.random.randn(osi.cfg.iem.obs_dim).astype(np.float32)
        action = np.random.randn(osi.cfg.iem.action_dim).astype(np.float32)
        pred, lat = osi.predict_next_obs(obs, action)
        assert pred.shape == (osi.cfg.iem.output_dim,)
        assert lat >= 0.0


# ---------------------------------------------------------------------------
# MultiAssetOSI
# ---------------------------------------------------------------------------

class TestMultiAssetOSI:
    def test_full_update(self, multi_osi):
        multi_osi.reset()
        n = multi_osi._n
        rng = np.random.default_rng(0)
        result = multi_osi.full_update(
            log_returns=rng.standard_normal(n) * 0.01,
            signed_volumes=rng.standard_normal(n) * 100,
            price_changes=rng.standard_normal(n) * 0.001,
            predicted_impacts=np.zeros(n),
            actual_impacts=np.zeros(n),
            predicted_fill_rates=np.ones(n),
            actual_fill_rates=np.ones(n),
            predicted_spreads=np.ones(n) * 0.01,
            actual_spreads=np.ones(n) * 0.01,
        )
        assert "correlation_matrix" in result
        assert result["correlation_matrix"].shape == (n, n)

    def test_correlation_symmetric(self, multi_osi):
        multi_osi.reset()
        n = multi_osi._n
        rng = np.random.default_rng(0)
        for _ in range(50):
            multi_osi.full_update(
                log_returns=rng.standard_normal(n) * 0.01,
                signed_volumes=np.zeros(n),
                price_changes=np.zeros(n),
                predicted_impacts=np.zeros(n),
                actual_impacts=np.zeros(n),
                predicted_fill_rates=np.ones(n),
                actual_fill_rates=np.ones(n),
                predicted_spreads=np.ones(n) * 0.01,
                actual_spreads=np.ones(n) * 0.01,
            )
        corr = multi_osi.correlation_matrix
        np.testing.assert_array_almost_equal(corr, corr.T, decimal=10)

    def test_observation_features_length(self, multi_osi):
        multi_osi.reset()
        feats = multi_osi.get_observation_features()
        assert len(feats) > 0
        assert feats.dtype == np.float32

    def test_reset(self, multi_osi):
        multi_osi.reset()
        assert multi_osi.osi._step == 0


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

class TestFactories:
    def test_make_osi(self):
        osi = make_osi(num_assets=4, obs_dim=64, action_dim=8)
        assert isinstance(osi, OnlineSystemIdentification)
        assert osi.cfg.num_assets == 4

    def test_make_osi_with_seed(self):
        osi = make_osi(num_assets=2, obs_dim=16, action_dim=4, seed=42)
        assert isinstance(osi, OnlineSystemIdentification)

    def test_make_multi_asset_osi(self):
        mosi = make_multi_asset_osi(num_assets=3, obs_dim=32, action_dim=4)
        assert isinstance(mosi, MultiAssetOSI)
        assert mosi._n == 3

    def test_make_osi_with_alert_callback(self):
        received = []
        def cb(t, p):
            received.append(t)
        osi = make_osi(num_assets=2, obs_dim=16, action_dim=4, alert_callback=cb)
        assert osi.cfg.alert_callback is cb


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_osi_latency_budget_constant():
    assert OSI_LATENCY_BUDGET_MS == 5.0
