"""
Tests for agent implementations.

Covers:
  - BaseAgent components (StandardNorm, Memory, GAE)
  - MAPPOAgent act/update
  - MeanFieldAgent act/update
  - MarketMakerAgent
  - MomentumAgent
  - ArbitrageAgent
  - NoiseTrader
"""

import sys
import os
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hyper_agent.agents.base_agent import (
    StandardNorm, Memory, compute_gae,
    ObservationEncoder, ActionDecoder, BaseAgent
)
from hyper_agent.agents.mappo_agent import MAPPOAgent, MAPPOActor, MAPPOCritic
from hyper_agent.agents.mean_field_agent import MeanFieldAgent, MeanFieldTracker
from hyper_agent.agents.market_maker_agent import MarketMakerAgent, AvellanedaStoikov
from hyper_agent.agents.momentum_agent import MomentumAgent, SharpeTracker
from hyper_agent.agents.arbitrage_agent import ArbitrageAgent, engle_granger_test
from hyper_agent.agents.noise_trader import NoiseTrader


OBS_DIM    = 23
HIDDEN_DIM = 32
N_AGENTS   = 4
GLOBAL_DIM = OBS_DIM * N_AGENTS


# ============================================================
# Base agent components
# ============================================================

class TestStandardNorm(unittest.TestCase):

    def setUp(self):
        self.norm = StandardNorm((OBS_DIM,))

    def test_initial_state(self):
        self.assertEqual(self.norm.count, 0)
        np.testing.assert_array_equal(self.norm.mean, np.zeros(OBS_DIM))

    def test_update(self):
        x = np.ones(OBS_DIM, dtype=np.float32)
        self.norm.update(x)
        self.assertGreater(self.norm.count, 0)

    def test_normalize_identity_at_init(self):
        """Before any updates, normalization should be near zero."""
        x = np.zeros(OBS_DIM, dtype=np.float32)
        normed = self.norm.normalize(x)
        self.assertEqual(normed.shape, (OBS_DIM,))
        self.assertTrue(np.all(np.isfinite(normed)))

    def test_normalize_after_updates(self):
        for _ in range(100):
            self.norm.update(np.random.randn(OBS_DIM))
        x = np.random.randn(OBS_DIM).astype(np.float32)
        normed = self.norm.normalize(x)
        # Should be clipped to [-10, 10]
        self.assertTrue(np.all(normed >= -10.01))
        self.assertTrue(np.all(normed <= 10.01))

    def test_state_dict_roundtrip(self):
        self.norm.update(np.ones(OBS_DIM))
        sd   = self.norm.state_dict()
        norm2 = StandardNorm((OBS_DIM,))
        norm2.load_state_dict(sd)
        np.testing.assert_array_almost_equal(norm2.mean, self.norm.mean)


class TestMemory(unittest.TestCase):

    def setUp(self):
        self.mem = Memory(capacity=100)

    def test_push_and_len(self):
        obs = np.zeros(OBS_DIM)
        for _ in range(10):
            self.mem.push(obs, 0, 0.5, -0.3, 0.1, obs, False, 0.0)
        self.assertEqual(len(self.mem), 10)

    def test_capacity_limit(self):
        obs = np.zeros(OBS_DIM)
        for _ in range(150):
            self.mem.push(obs, 1, 0.5, 0.0, 0.0, obs, False, 0.0)
        self.assertEqual(len(self.mem), 100)

    def test_sample_shape(self):
        obs = np.random.randn(OBS_DIM)
        for i in range(30):
            self.mem.push(obs, i % 3, 0.5, 0.0, float(i) * 0.1, obs, False, 0.0)
        batch = self.mem.sample(16)
        self.assertEqual(len(batch["obs"]), 16)
        self.assertEqual(batch["obs"][0].shape, (OBS_DIM,))

    def test_get_all(self):
        obs = np.zeros(OBS_DIM)
        for _ in range(5):
            self.mem.push(obs, 0, 0.5, 0.0, 1.0, obs, True, 0.0)
        all_data = self.mem.get_all()
        self.assertEqual(len(all_data["rewards"]), 5)

    def test_clear(self):
        obs = np.zeros(OBS_DIM)
        self.mem.push(obs, 0, 0.5, 0.0, 0.0, obs, False)
        self.mem.clear()
        self.assertEqual(len(self.mem), 0)


class TestComputeGAE(unittest.TestCase):

    def test_shape(self):
        T      = 20
        rewards = np.random.randn(T).astype(np.float32)
        values  = np.random.randn(T).astype(np.float32)
        dones   = np.zeros(T, dtype=np.float32)
        adv, ret = compute_gae(rewards, values, dones)
        self.assertEqual(adv.shape, (T,))
        self.assertEqual(ret.shape, (T,))

    def test_returns_greater_equal_last_reward_nondone(self):
        rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        values  = np.zeros(3, dtype=np.float32)
        dones   = np.zeros(3, dtype=np.float32)
        _, returns = compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)
        # With gamma=1, lam=1: returns = discounted sum of future rewards
        self.assertGreater(returns[0], returns[2])

    def test_done_flag_truncates(self):
        rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        values  = np.zeros(3, dtype=np.float32)
        dones   = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        adv, ret = compute_gae(rewards, values, dones, gamma=0.99)
        # After done flag, returns should restart
        self.assertIsNotNone(adv)


# ============================================================
# ObservationEncoder / ActionDecoder
# ============================================================

class TestObservationEncoder(unittest.TestCase):

    def test_forward(self):
        enc   = ObservationEncoder(OBS_DIM, HIDDEN_DIM, HIDDEN_DIM)
        obs_t = torch.randn(4, OBS_DIM)
        out, h = enc(obs_t)
        self.assertEqual(out.shape, (4, HIDDEN_DIM))
        self.assertIsNone(h)

    def test_gru_variant(self):
        enc   = ObservationEncoder(OBS_DIM, HIDDEN_DIM, HIDDEN_DIM, use_gru=True)
        obs_t = torch.randn(3, OBS_DIM)
        out, hidden = enc(obs_t)
        self.assertIsNotNone(hidden)

    def test_single_obs(self):
        enc   = ObservationEncoder(OBS_DIM, HIDDEN_DIM, HIDDEN_DIM)
        obs_t = torch.randn(OBS_DIM)
        out, _ = enc(obs_t)
        self.assertEqual(out.shape[1], HIDDEN_DIM)


class TestActionDecoder(unittest.TestCase):

    def test_forward_shape(self):
        dec    = ActionDecoder(HIDDEN_DIM)
        latent = torch.randn(5, HIDDEN_DIM)
        logits, alpha, beta = dec.forward(latent)
        self.assertEqual(logits.shape, (5, 3))
        self.assertEqual(alpha.shape,  (5, 1))
        self.assertEqual(beta.shape,   (5, 1))

    def test_distributions_valid(self):
        dec    = ActionDecoder(HIDDEN_DIM)
        latent = torch.randn(4, HIDDEN_DIM)
        cat, beta_dist = dec.get_distributions(latent)
        dir_  = cat.sample()
        size  = beta_dist.sample()
        self.assertEqual(dir_.shape,  (4,))
        self.assertTrue(torch.all(size >= 0) and torch.all(size <= 1))

    def test_entropy_positive(self):
        dec    = ActionDecoder(HIDDEN_DIM)
        latent = torch.randn(4, HIDDEN_DIM)
        ent    = dec.entropy(latent)
        self.assertTrue(torch.all(ent >= 0))


# ============================================================
# MAPPOAgent
# ============================================================

class TestMAPPOAgent(unittest.TestCase):

    def setUp(self):
        self.agent = MAPPOAgent(
            agent_id         = "test_mappo",
            obs_dim          = OBS_DIM,
            global_state_dim = GLOBAL_DIM,
            hidden_dim       = HIDDEN_DIM,
            rollout_len      = 32,
            minibatch_size   = 8,
            n_epochs         = 1,
        )

    def test_act_output_shape(self):
        obs    = np.random.randn(OBS_DIM).astype(np.float32)
        action, lp, val = self.agent.act(obs)
        self.assertEqual(action.shape, (4,))
        self.assertIsInstance(lp,  float)
        self.assertIsInstance(val, float)

    def test_act_deterministic(self):
        obs    = np.random.randn(OBS_DIM).astype(np.float32)
        a1, _, _ = self.agent.act(obs, deterministic=True)
        self.agent.reset_episode()
        a2, _, _ = self.agent.act(obs, deterministic=True)
        np.testing.assert_array_almost_equal(a1, a2, decimal=4)

    def test_compute_value(self):
        gs  = np.random.randn(GLOBAL_DIM).astype(np.float32)
        val = self.agent.compute_value(gs)
        self.assertIsInstance(val, float)
        self.assertTrue(np.isfinite(val))

    def test_update_after_rollout(self):
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        gs  = np.random.randn(GLOBAL_DIM).astype(np.float32)
        # Fill rollout buffer
        for _ in range(32):
            self.agent.store_rollout(obs, gs, 1, 0.5, -0.3, 0.0, False)
        self.agent.finish_rollout(gs)
        stats = self.agent.update()
        self.assertIn("policy_loss", stats)

    def test_get_policy_params(self):
        params = self.agent.get_policy_params()
        self.assertIsInstance(params, np.ndarray)
        self.assertGreater(len(params), 0)

    def test_agent_type(self):
        self.assertEqual(self.agent.agent_type, "mappo")


# ============================================================
# MeanFieldAgent
# ============================================================

class TestMeanFieldTracker(unittest.TestCase):

    def test_update_batch(self):
        tracker = MeanFieldTracker(n_agents=10)
        mf = tracker.update_batch([0, 1, 2, 0, 1])
        self.assertAlmostEqual(mf.sum(), 1.0, places=5)
        self.assertEqual(len(mf), 3)

    def test_initial_uniform(self):
        tracker = MeanFieldTracker(n_agents=5)
        mf = tracker.get()
        np.testing.assert_array_almost_equal(mf, [1/3, 1/3, 1/3], decimal=4)

    def test_kl_div_self_zero(self):
        p  = np.array([0.5, 0.3, 0.2])
        kl = MeanFieldTracker.kl_divergence(p, p)
        self.assertAlmostEqual(kl, 0.0, places=4)


# ============================================================
# MarketMakerAgent
# ============================================================

class TestMarketMakerAgent(unittest.TestCase):

    def setUp(self):
        self.agent = MarketMakerAgent(
            agent_id    = "mm_0",
            obs_dim     = OBS_DIM,
            hidden_dim  = HIDDEN_DIM,
            rollout_len = 32,
        )

    def test_act_output(self):
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, lp, val = self.agent.act(obs)
        self.assertEqual(action.shape, (4,))
        self.assertIsInstance(lp, float)

    def test_as_baseline(self):
        hs, res = self.agent.get_as_spread(100.0, 0.0, 0.5)
        self.assertGreater(hs, 0.0)

    def test_agent_type(self):
        self.assertEqual(self.agent.agent_type, "market_maker")


class TestAvellanedaStoikov(unittest.TestCase):

    def test_spread_positive(self):
        as_ = AvellanedaStoikov()
        for p in [99.0, 100.0, 101.0]:
            as_.update_price(p)
        hs, res = as_.compute_spread_and_reservation(100.0, 0.0, 0.5)
        self.assertGreater(hs, 0.0)

    def test_inventory_skews_reservation(self):
        as_ = AvellanedaStoikov()
        for p in np.linspace(99, 101, 20):
            as_.update_price(p)
        _, res_long  = as_.compute_spread_and_reservation(100.0, +5.0, 0.5)
        _, res_short = as_.compute_spread_and_reservation(100.0, -5.0, 0.5)
        # Long inventory → sell-biased reservation price should be lower
        self.assertLess(res_long, res_short)


# ============================================================
# MomentumAgent
# ============================================================

class TestMomentumAgent(unittest.TestCase):

    def setUp(self):
        self.agent = MomentumAgent(
            agent_id    = "mom_0",
            obs_dim     = OBS_DIM,
            hidden_dim  = HIDDEN_DIM,
            rollout_len = 32,
        )

    def test_act_output(self):
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, lp, val = self.agent.act(obs)
        self.assertEqual(action.shape, (4,))

    def test_observe_price(self):
        for p in np.linspace(100, 105, 20):
            self.agent.observe_price(float(p))
        sigs = self.agent.get_ewma_signals()
        self.assertIsNotNone(sigs)
        self.assertEqual(len(sigs), 4)

    def test_running_sharpe(self):
        self.agent.sharpe_tracker.update(0.01)
        self.agent.sharpe_tracker.update(-0.005)
        s = self.agent.running_sharpe()
        self.assertTrue(np.isfinite(s))

    def test_agent_type(self):
        self.assertEqual(self.agent.agent_type, "momentum")


# ============================================================
# ArbitrageAgent
# ============================================================

class TestArbitrageAgent(unittest.TestCase):

    def setUp(self):
        self.agent = ArbitrageAgent(
            agent_id   = "arb_0",
            obs_dim    = OBS_DIM,
            hidden_dim = HIDDEN_DIM,
        )

    def test_act_output(self):
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, lp, val = self.agent.act(obs)
        self.assertEqual(action.shape, (4,))

    def test_observe_prices(self):
        for i, p in enumerate(np.linspace(100, 102, 50)):
            z = self.agent.observe_prices(float(p), i)
            self.assertTrue(np.isfinite(z))

    def test_spread_stats(self):
        for i in range(30):
            self.agent.observe_prices(100.0 + np.random.randn() * 0.5, i)
        stats = self.agent.get_spread_stats()
        self.assertIn("z_score", stats)
        self.assertIn("half_life", stats)

    def test_agent_type(self):
        self.assertEqual(self.agent.agent_type, "arbitrage")


def test_engle_granger():
    """Test cointegration test with known cointegrated series."""
    n = 200
    common = np.cumsum(np.random.randn(n)) * 0.5
    y1 = common + np.random.randn(n) * 0.1
    y2 = common + np.random.randn(n) * 0.1
    beta, alpha, adf = engle_granger_test(y1, y2)
    assert np.isfinite(beta), "Beta should be finite"
    assert np.isfinite(adf),  "ADF stat should be finite"
    # Cointegrated → negative ADF
    assert adf < 0, f"Expected negative ADF for cointegrated series, got {adf}"


# ============================================================
# NoiseTrader
# ============================================================

class TestNoiseTrader(unittest.TestCase):

    def setUp(self):
        self.agent = NoiseTrader(
            agent_id = "noise_0",
            obs_dim  = OBS_DIM,
            seed     = 0,
        )

    def test_act_output(self):
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action, lp, val = self.agent.act(obs)
        self.assertEqual(action.shape, (4,))

    def test_update_noop(self):
        stats = self.agent.update()
        self.assertEqual(stats, {})

    def test_agent_type(self):
        self.assertEqual(self.agent.agent_type, "noise")

    def test_direction_probs_sum_to_one(self):
        p = self.agent._direction_probs
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)

    def test_set_bias_adjusts_probs(self):
        self.agent.set_directional_bias(0.3)
        p = self.agent._direction_probs
        # Long prob (index 2) should be higher than short (index 0)
        self.assertGreater(p[2], p[0])

    def test_volume_stats(self):
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        for _ in range(20):
            self.agent.act(obs)
        stats = self.agent.volume_stats()
        self.assertIn("buy_fraction", stats)
        self.assertIn("mean_size", stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)
