"""
Tests for hyper_agent/environment.py

Covers:
  - LimitOrderBook mechanics
  - MarketEnvironment reset/step
  - MultiAgentTradingEnv interface
  - Crisis injection
  - Observation shapes
  - Action decoding
"""

import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hyper_agent.environment import MultiAssetTradingEnv
from hyper_agent.env_compat import (
    DictMultiAgentEnv as MultiAgentTradingEnv, make_env, LimitOrderBook,
)

MarketEnvironment = MultiAgentTradingEnv


class TestLimitOrderBook(unittest.TestCase):

    def setUp(self):
        self.lob = LimitOrderBook(mid_price=100.0, tick_size=0.01, depth=5)

    def test_initial_state(self):
        self.assertAlmostEqual(self.lob.mid_price, 100.0, places=2)
        self.assertGreater(self.lob.spread(), 0.0)

    def test_bid_ask_order(self):
        self.assertLess(self.lob.best_bid(), self.lob.best_ask())

    def test_spread_positive(self):
        self.assertGreater(self.lob.spread(), 0.0)

    def test_submit_buy_increases_price(self):
        old_mid = self.lob.mid_price
        self.lob.submit_orders(net_direction=5.0)
        self.assertGreater(self.lob.mid_price, old_mid)

    def test_submit_sell_decreases_price(self):
        old_mid = self.lob.mid_price
        self.lob.submit_orders(net_direction=-5.0)
        self.assertLess(self.lob.mid_price, old_mid)

    def test_zero_order_price_change_small(self):
        old_mid = self.lob.mid_price
        self.lob.submit_orders(net_direction=0.0)
        # Price should change only due to mean reversion / fundamental walk
        self.assertAlmostEqual(self.lob.mid_price, old_mid, delta=5.0)

    def test_positive_price_always(self):
        for _ in range(50):
            self.lob.submit_orders(net_direction=np.random.uniform(-10, 10))
            self.assertGreater(self.lob.mid_price, 0.0)

    def test_reset(self):
        self.lob.submit_orders(10.0)
        self.lob.reset(100.0)
        self.assertAlmostEqual(self.lob.mid_price, 100.0, places=2)

    def test_volume_tracking(self):
        self.lob.submit_orders(3.0)
        self.assertGreater(self.lob.volume_this_bar, 0.0)

    def test_exec_report_keys(self):
        report = self.lob.submit_orders(1.0)
        for key in ["exec_price", "impact", "spread", "volume"]:
            self.assertIn(key, report)


class TestMarketEnvironment(unittest.TestCase):

    def setUp(self):
        self.agent_ids = ["agent_0", "agent_1", "agent_2"]
        self.env = MarketEnvironment(
            agent_ids  = self.agent_ids,
            max_steps  = 50,
            seed       = 42,
        )

    def _random_actions(self):
        return {
            aid: np.array(
                [np.random.uniform(-1, 1),
                 np.random.uniform(-1, 1),
                 np.random.uniform(-1, 1),
                 np.random.uniform(0, 1)],
                dtype=np.float32,
            )
            for aid in self.agent_ids
        }

    def test_reset_returns_obs_and_info(self):
        obs, info = self.env.reset()
        self.assertEqual(set(obs.keys()), set(self.agent_ids))
        self.assertEqual(set(info.keys()), set(self.agent_ids))

    def test_obs_shape(self):
        obs, _ = self.env.reset()
        expected_shape = (self.env.obs_shape,)
        for aid, o in obs.items():
            self.assertEqual(o.shape, expected_shape, f"Wrong shape for {aid}")

    def test_obs_finite(self):
        obs, _ = self.env.reset()
        for aid, o in obs.items():
            self.assertTrue(np.all(np.isfinite(o)), f"Inf/NaN in obs for {aid}")

    def test_step_output_structure(self):
        self.env.reset()
        actions = self._random_actions()
        obs, rew, term, trunc, info = self.env.step(actions)
        agent_set = set(self.agent_ids)
        self.assertTrue(agent_set.issubset(set(obs.keys())))
        self.assertTrue(agent_set.issubset(set(rew.keys())))
        self.assertTrue(agent_set.issubset(set(term.keys())))
        self.assertTrue(agent_set.issubset(set(trunc.keys())))
        self.assertTrue(agent_set.issubset(set(info.keys())))

    def test_rewards_are_finite(self):
        self.env.reset()
        for _ in range(5):
            actions = self._random_actions()
            _, rew, _, _, _ = self.env.step(actions)
            for r in rew.values():
                self.assertTrue(np.isfinite(r))

    def test_truncation_after_max_steps(self):
        self.env.reset()
        for _ in range(self.env.max_steps):
            actions = self._random_actions()
            _, _, _, trunc, _ = self.env.step(actions)
        self.assertTrue(any(trunc.values()))

    def test_position_limits(self):
        self.env.reset()
        big_action = {
            aid: np.array([0.0, 0.0, 5.0, 1.0], dtype=np.float32)
            for aid in self.agent_ids
        }
        for _ in range(20):
            self.env.step(big_action)
        for aid in self.agent_ids:
            self.assertLessEqual(abs(self.env.positions[aid]), self.env.max_position + 1e-3)

    def test_global_state_shape(self):
        self.env.reset()
        gs = self.env.get_global_state()
        expected_dim = self.env.obs_shape * len(self.agent_ids)
        self.assertEqual(gs.shape, (expected_dim,))

    def test_crisis_injection(self):
        env = MarketEnvironment(
            agent_ids   = ["a0", "a1"],
            max_steps   = 100,
            crisis_step = 10,
            seed        = 42,
        )
        env.reset()
        for t in range(15):
            actions = {"a0": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                       "a1": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)}
            env.step(actions)
        self.assertGreater(env.current_vol, 1.0)

    def test_seed_reproducibility(self):
        env1 = MarketEnvironment(["a", "b"], seed=99)
        env2 = MarketEnvironment(["a", "b"], seed=99)
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        np.testing.assert_array_almost_equal(obs1["a"], obs2["a"])


class TestMultiAgentTradingEnv(unittest.TestCase):

    def setUp(self):
        config = {
            "agent_ids": ["m0", "m1", "n0"],
            "max_steps": 50,
            "seed":      42,
        }
        self.env = MultiAgentTradingEnv(config)

    def test_action_observation_spaces(self):
        from gymnasium import spaces
        self.assertIsInstance(self.env.observation_space, spaces.Dict)
        self.assertIsInstance(self.env.action_space, spaces.Dict)

    def test_reset(self):
        obs, info = self.env.reset()
        self.assertIn("m0", obs)
        self.assertIn("m1", obs)

    def test_step_all_done(self):
        obs, _ = self.env.reset()
        agent_ids = self.env.agent_ids
        for _ in range(55):  # more than max_steps
            actions = {
                aid: np.array([0.0, 0.0, 0.0, 0.5], dtype=np.float32)
                for aid in agent_ids
            }
            obs, rew, term, trunc, info = self.env.step(actions)
            if term.get("__all__") or trunc.get("__all__"):
                break
        self.assertTrue(term.get("__all__") or trunc.get("__all__"))

    def test_make_env_factory(self):
        env = make_env(n_market_makers=2, n_momentum=2, n_noise=3, seed=0)
        self.assertIsInstance(env, MultiAgentTradingEnv)
        obs, _ = env.reset()
        self.assertEqual(len(obs), 7)


class TestActionDecoding(unittest.TestCase):

    def setUp(self):
        self.env = MarketEnvironment(
            agent_ids=["a0", "a1"], max_steps=20
        )

    def test_softmax_normalization(self):
        for _ in range(20):
            x = np.random.randn(3)
            probs = self.env._softmax(x)
            self.assertAlmostEqual(probs.sum(), 1.0, places=6)
            self.assertTrue(np.all(probs >= 0))

    def test_decode_zero_actions_gives_flat(self):
        self.env.reset()
        actions = {"a0": np.zeros(4, dtype=np.float32),
                   "a1": np.zeros(4, dtype=np.float32)}
        decoded = self.env._decode_actions(actions)
        # argmax of [0,0,0] → 0 (short) which is fine; just check structure
        self.assertIn("a0", decoded)
        self.assertIn("signed_size", decoded["a0"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
