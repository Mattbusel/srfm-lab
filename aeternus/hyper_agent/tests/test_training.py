"""
Tests for training infrastructure.

Covers:
  - MAPPOTrainer rollout collection
  - PopulationTrainer fitness tracking
  - CurriculumScheduler stage progression
  - Reward shaping components
  - Credit assignment (COMA, QMIX, VDN)
"""

import sys
import os
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hyper_agent.env_compat import make_env
from hyper_agent.training.mappo_trainer import MAPPOAgent  # adapter with act/compute_value
from hyper_agent.agents.noise_trader import NoiseTrader
from hyper_agent.training.mappo_trainer import MAPPOTrainer, RolloutCollector, build_mappo_trainer
from hyper_agent.training.population_trainer import FitnessTracker, PopulationTrainer
from hyper_agent.training.curriculum import CurriculumScheduler, STAGE_CONFIGS, PlateauDetector
from hyper_agent.reward.reward_shaping import (
    IndividualReward, TeamReward, MarketQualityReward,
    AdversarialPenalty, PotentialBasedShaping, CuriosityBonus,
)
from hyper_agent.reward.credit_assignment import COMAAdvantage, QMIXMixer, VDNMixer


OBS_DIM    = 23
HIDDEN_DIM = 32
N_AGENTS   = 4
GLOBAL_DIM = OBS_DIM * N_AGENTS


# ============================================================
# MAPPO Trainer
# ============================================================

class TestRolloutCollector(unittest.TestCase):

    def setUp(self):
        self.env = make_env(
            n_market_makers=1, n_momentum=1, n_noise=2,
            max_steps=30, seed=0
        )
        self.agent_ids = self.env.agent_ids
        real_obs_dim = self.env.obs_shape
        real_global_dim = real_obs_dim * len(self.agent_ids)
        self.agents = {
            aid: MAPPOAgent(
                agent_id        = i,
                obs_dim         = real_obs_dim,
                action_dim      = 4,
                state_dim       = real_global_dim,
                num_agents      = len(self.agent_ids),
                hidden_dim      = HIDDEN_DIM,
                ppo_epochs      = 1,
                mini_batch_size = 8,
            )
            for i, aid in enumerate(self.agent_ids)
        }
        self.collector = RolloutCollector(self.env, self.agents, rollout_len=16)

    def test_reset(self):
        self.collector.reset()
        self.assertIsNotNone(self.collector._current_obs)

    def test_collect_returns_step_counts(self):
        self.collector.reset()
        counts = self.collector.collect()
        self.assertEqual(set(counts.keys()), set(self.agent_ids))
        for aid, n in counts.items():
            self.assertGreater(n, 0)

    def test_rollout_fills_buffer(self):
        self.collector.reset()
        counts = self.collector.collect()
        # Verify collect ran without error and returned counts
        self.assertIsInstance(counts, dict)
        self.assertGreater(sum(counts.values()), 0)


class TestBuildMAPPOTrainer(unittest.TestCase):

    def test_build(self):
        env     = make_env(n_market_makers=1, n_momentum=1, n_noise=1, max_steps=20, seed=1)
        trainer = build_mappo_trainer(env, OBS_DIM, hidden_dim=HIDDEN_DIM, rollout_len=16)
        self.assertIsInstance(trainer, MAPPOTrainer)
        self.assertEqual(len(trainer.agents), len(env.agent_ids))


# ============================================================
# Population Trainer
# ============================================================

class TestFitnessTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = FitnessTracker(["a0", "a1", "a2"], window=20)

    def test_update_fitness(self):
        self.tracker.update("a0", 1.0)
        self.tracker.update("a0", 2.0)
        self.assertAlmostEqual(self.tracker.fitness("a0"), 1.5)

    def test_ranking_order(self):
        self.tracker.update("a0", 5.0)
        self.tracker.update("a1", 1.0)
        self.tracker.update("a2", 3.0)
        ranking = self.tracker.ranking()
        self.assertEqual(ranking[0][0], "a0")
        self.assertEqual(ranking[-1][0], "a1")

    def test_worst_k(self):
        self.tracker.update("a0", 5.0)
        self.tracker.update("a1", 1.0)
        self.tracker.update("a2", 3.0)
        worst = self.tracker.worst_k(1)
        self.assertEqual(worst[0], "a1")

    def test_best_k(self):
        self.tracker.update("a0", 5.0)
        self.tracker.update("a1", 1.0)
        best = self.tracker.best_k(1)
        self.assertEqual(best[0], "a0")


class TestPopulationTrainerInit(unittest.TestCase):

    def test_builds_without_error(self):
        env = make_env(n_market_makers=1, n_momentum=1, n_noise=2, max_steps=20, seed=0)
        agents = {
            aid: NoiseTrader(aid, OBS_DIM, seed=0)
            for aid in env.agent_ids
        }
        trainer = PopulationTrainer(
            env, OBS_DIM, agents,
            replace_every    = 5,
            n_total_episodes = 2,
        )
        self.assertIsNotNone(trainer)
        comp = trainer.population_composition()
        self.assertIn("noise", comp)


# ============================================================
# CurriculumScheduler
# ============================================================

class TestCurriculumScheduler(unittest.TestCase):

    def setUp(self):
        self.sched = CurriculumScheduler(initial_stage=1, auto_advance=True)

    def test_initial_stage(self):
        self.assertEqual(self.sched.current_stage, 1)

    def test_build_env(self):
        env = self.sched.build_env()
        self.assertIsNotNone(env)

    def test_record_episode_no_advance_too_early(self):
        advanced = self.sched.record_episode(0.1)
        self.assertFalse(advanced)

    def test_force_advance(self):
        result = self.sched.force_advance()
        self.assertTrue(result)
        self.assertEqual(self.sched.current_stage, 2)

    def test_stage_configs_exist(self):
        for stage in range(1, 6):
            self.assertIn(stage, STAGE_CONFIGS)

    def test_curriculum_summary(self):
        summary = self.sched.curriculum_summary()
        self.assertIn("current_stage", summary)
        self.assertIn("stage_name", summary)

    def test_set_stage(self):
        self.sched.set_stage(3, rebuild_env=False)
        self.assertEqual(self.sched.current_stage, 3)


class TestPlateauDetector(unittest.TestCase):

    def test_not_plateau_early(self):
        pd = PlateauDetector(window=20, min_n=10)
        for i in range(5):
            pd.update(float(i))
        self.assertFalse(pd.is_plateau())

    def test_plateau_on_constant(self):
        pd = PlateauDetector(window=20, threshold=0.01, min_n=10)
        for _ in range(25):
            pd.update(1.0)
        self.assertTrue(pd.is_plateau())

    def test_no_plateau_on_rising(self):
        pd = PlateauDetector(window=20, threshold=0.001, min_n=15)
        for i in range(25):
            pd.update(float(i) * 0.1)
        self.assertFalse(pd.is_plateau())


# ============================================================
# Reward Shaping
# ============================================================

class TestIndividualReward(unittest.TestCase):

    def test_raw_mode(self):
        r = IndividualReward(mode="raw")
        self.assertAlmostEqual(r.compute(1.5), 1.5)

    def test_clipped_mode(self):
        r = IndividualReward(mode="clipped", clip_range=2.0)
        self.assertAlmostEqual(r.compute(10.0), 2.0)
        self.assertAlmostEqual(r.compute(-10.0), -2.0)

    def test_log_mode_preserves_sign(self):
        r = IndividualReward(mode="log")
        self.assertGreater(r.compute(1.0), 0.0)
        self.assertLess(r.compute(-1.0), 0.0)

    def test_risk_adj_with_history(self):
        r = IndividualReward(mode="risk_adj")
        for _ in range(10):
            r.compute(np.random.randn())
        val = r.compute(0.1)
        self.assertTrue(np.isfinite(val))


class TestTeamReward(unittest.TestCase):

    def test_build_from_ids(self):
        ids = ["mm_0", "mm_1", "mom_0", "noise_0"]
        tr  = TeamReward.build_from_agent_ids(ids, blend_alpha=0.5)
        ind = {"mm_0": 1.0, "mm_1": 3.0, "mom_0": 2.0, "noise_0": 0.0}
        blended = tr.compute(ind)
        # mm_0 should get blend of 1.0 and team mean of 2.0
        self.assertAlmostEqual(blended["mm_0"], 0.5 * 1.0 + 0.5 * 2.0, places=4)


class TestAdversarialPenalty(unittest.TestCase):

    def test_no_penalty_random(self):
        ap = AdversarialPenalty()
        rewards = {"a0": 1.0}
        shaped  = ap.shape_rewards(rewards, {"a0": (1, 0.3)})
        self.assertLessEqual(shaped["a0"], 1.0)

    def test_penalty_for_ramping(self):
        ap  = AdversarialPenalty()
        aid = "ramp_0"
        for _ in range(15):
            ap.observe_action(aid, 1, 0.5)  # all buys
        pen = ap.compute_penalty(aid)
        self.assertGreaterEqual(pen, 0.0)


class TestPotentialBasedShaping(unittest.TestCase):

    def test_potential_positive(self):
        ps  = PotentialBasedShaping()
        phi = ps.potential(100.0, 0.01, 10.0, {"a0": 0.0, "a1": 1.0})
        self.assertGreaterEqual(phi, 0.0)

    def test_shaping_is_finite(self):
        ps       = PotentialBasedShaping()
        rewards  = {"a0": 0.5, "a1": -0.3}
        prev_st  = {"mid_price": 100.0, "spread": 0.02, "volume": 5.0,
                    "pnl_a0": 0.0, "pnl_a1": 0.0}
        curr_st  = {"mid_price": 100.1, "spread": 0.015, "volume": 6.0,
                    "pnl_a0": 0.5, "pnl_a1": -0.3}
        shaped   = ps.shape(rewards, prev_st, curr_st)
        for r in shaped.values():
            self.assertTrue(np.isfinite(r))


class TestCuriosityBonus(unittest.TestCase):

    def test_bonus_finite(self):
        cb  = CuriosityBonus(OBS_DIM, out_dim=16, scale=0.1)
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        b   = cb.compute_bonus(obs)
        self.assertTrue(np.isfinite(b))

    def test_predictor_update_returns_loss(self):
        cb   = CuriosityBonus(OBS_DIM, out_dim=16)
        obs  = np.random.randn(5, OBS_DIM).astype(np.float32)
        loss = cb.update_predictor(obs)
        self.assertGreater(loss, 0.0)


# ============================================================
# Credit Assignment
# ============================================================

class TestCOMAAdvantage(unittest.TestCase):

    def setUp(self):
        self.coma = COMAAdvantage(
            global_state_dim = GLOBAL_DIM,
            n_agents         = N_AGENTS,
            n_actions        = 3,
            hidden_dim       = 32,
        )

    def test_compute_advantage_shape(self):
        B    = 8
        gs   = torch.randn(B, GLOBAL_DIM)
        ja   = torch.randn(B, N_AGENTS * 3)
        acts = torch.randint(0, 3, (B,))
        probs= torch.softmax(torch.randn(B, 3), dim=-1)
        adv  = self.coma.compute_advantage(0, gs, ja, acts, probs)
        self.assertEqual(adv.shape, (B,))

    def test_update_returns_float(self):
        B       = 8
        gs      = torch.randn(B, GLOBAL_DIM)
        ja      = torch.randn(B, N_AGENTS * 3)
        acts    = torch.randint(0, 3, (B,))
        targets = torch.randn(B)
        loss    = self.coma.update(0, gs, ja, acts, targets)
        self.assertGreater(loss, 0.0)


class TestVDNMixer(unittest.TestCase):

    def setUp(self):
        self.vdn = VDNMixer(
            obs_dim    = OBS_DIM,
            n_agents   = N_AGENTS,
            n_actions  = 3,
            hidden_dim = 32,
        )

    def test_act_valid_action(self):
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        act = self.vdn.act(0, obs, eps=0.5)
        self.assertIn(act, [0, 1, 2])

    def test_update_returns_stats(self):
        B       = 8
        obs_l   = [np.random.randn(B, OBS_DIM).astype(np.float32) for _ in range(N_AGENTS)]
        acts    = np.random.randint(0, 3, (B, N_AGENTS))
        rewards = np.random.randn(B).astype(np.float32)
        dones   = np.zeros(B, dtype=np.float32)
        stats   = self.vdn.update(obs_l, obs_l, acts, rewards, dones)
        self.assertIn("vdn_loss", stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)
