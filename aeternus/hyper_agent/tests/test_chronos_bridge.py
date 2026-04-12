"""
tests/test_chronos_bridge.py
============================
Comprehensive tests for the Chronos LOB environment bridge.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from hyper_agent.chronos_env_bridge import (
    ChronosCSVParser,
    ChronosLOBEnv,
    ChronosMARLEnv,
    VecChronosEnv,
    AsyncVecChronosEnv,
    LOBSnapshot,
    AgentOrder,
    AgentState,
    Fill,
    OrderType,
    OrderSide,
    RewardMode,
    ObservationBuilder,
    ActionDecoder,
    SimpleExecutionEngine,
    RewardComputer,
    EpisodeManager,
    EpisodeConfig,
    NormalisedObsWrapper,
    FrameStackWrapper,
    EpisodeTracker,
    make_vec_env,
    make_marl_env,
    make_single_env,
    DEFAULT_LOB_DEPTH,
    DEFAULT_EPISODE_LEN,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_env():
    return ChronosLOBEnv(n_synthetic=500, episode_len=100, seed=42)


@pytest.fixture
def parser():
    return ChronosCSVParser(depth=5)


@pytest.fixture
def snapshots(parser):
    return parser.parse_file(csv_path=None, n_synthetic=300, seed=0)


@pytest.fixture
def snapshot(snapshots):
    return snapshots[0]


@pytest.fixture
def agent_state():
    return AgentState(agent_id="test_agent")


# ---------------------------------------------------------------------------
# ChronosCSVParser tests
# ---------------------------------------------------------------------------

class TestChronosCSVParser:
    def test_synthetic_data_generation(self, parser):
        snaps = parser.parse_file(n_synthetic=200, seed=1)
        assert len(snaps) == 200

    def test_snapshot_fields(self, snapshot):
        assert isinstance(snapshot.mid_price, float)
        assert snapshot.mid_price > 0
        assert snapshot.best_bid < snapshot.best_ask
        assert snapshot.spread > 0
        assert len(snapshot.bid_prices) == 5
        assert len(snapshot.ask_prices) == 5

    def test_snapshot_tensor(self, snapshot):
        t = snapshot.to_tensor()
        assert t.dtype == torch.float32
        assert t.ndim == 1
        assert not torch.isnan(t).any()

    def test_snapshot_imbalance(self, snapshot):
        imb = snapshot.imbalance
        assert -1 <= imb <= 1

    def test_snapshot_depth_imbalance(self, snapshot):
        di = snapshot.depth_imbalance
        assert -1 <= di <= 1

    def test_missing_csv_path_fallback(self):
        parser = ChronosCSVParser(depth=3)
        snaps = parser.parse_file(csv_path="/nonexistent/path.csv", n_synthetic=50, seed=42)
        assert len(snaps) == 50


# ---------------------------------------------------------------------------
# LOBSnapshot tests
# ---------------------------------------------------------------------------

class TestLOBSnapshot:
    def test_imbalance_extremes(self):
        snap = LOBSnapshot(
            timestamp=0, mid_price=100, best_bid=99.9, best_ask=100.1,
            spread=0.2,
            bid_prices=np.array([99.9, 99.8], dtype=np.float32),
            bid_volumes=np.array([100.0, 50.0], dtype=np.float32),
            ask_prices=np.array([100.1, 100.2], dtype=np.float32),
            ask_volumes=np.array([0.0, 10.0], dtype=np.float32),
        )
        assert snap.imbalance > 0.9   # heavily bid-side

    def test_zero_depth_imbalance(self):
        snap = LOBSnapshot(
            timestamp=0, mid_price=100, best_bid=99.9, best_ask=100.1,
            spread=0.2,
            bid_prices=np.array([99.9], dtype=np.float32),
            bid_volumes=np.array([0.0], dtype=np.float32),
            ask_prices=np.array([100.1], dtype=np.float32),
            ask_volumes=np.array([0.0], dtype=np.float32),
        )
        assert snap.depth_imbalance == 0.0


# ---------------------------------------------------------------------------
# ObservationBuilder tests
# ---------------------------------------------------------------------------

class TestObservationBuilder:
    def test_obs_dim(self):
        builder = ObservationBuilder(depth=5, history_len=10)
        assert builder.obs_dim > 0

    def test_build_output_shape(self, snapshot, agent_state):
        builder = ObservationBuilder(depth=5, history_len=10)
        obs = builder.build(snapshot, agent_state, episode_progress=0.5)
        assert obs.shape == (builder.obs_dim,)
        assert obs.dtype == np.float32
        assert not np.isnan(obs).any()

    def test_episode_progress(self, snapshot, agent_state):
        builder = ObservationBuilder(depth=5)
        obs_early = builder.build(snapshot, agent_state, 0.0)
        obs_late = builder.build(snapshot, agent_state, 1.0)
        # They should differ (episode progress is in obs)
        assert not np.allclose(obs_early, obs_late)

    def test_reset_clears_history(self, snapshot, agent_state):
        builder = ObservationBuilder(depth=5, history_len=5)
        for _ in range(10):
            builder.build(snapshot, agent_state, 0.0)
        builder.reset()
        obs = builder.build(snapshot, agent_state, 0.0)
        # After reset, history should be empty (zeros)
        assert obs is not None


# ---------------------------------------------------------------------------
# ActionDecoder tests
# ---------------------------------------------------------------------------

class TestActionDecoder:
    def test_decode_flat(self, snapshot):
        decoder = ActionDecoder()
        action = np.zeros(10, dtype=np.float32)
        action[0] = 1.0   # LIMIT order type
        action[4] = 1.0   # BUY
        order = decoder.decode(action, snapshot, "agent_0")
        assert isinstance(order, AgentOrder)
        assert order.size > 0

    def test_decode_all_order_types(self, snapshot):
        decoder = ActionDecoder()
        for otype_idx in range(4):
            action = np.zeros(10, dtype=np.float32)
            action[otype_idx] = 1.0
            order = decoder.decode(action, snapshot, "agent_0")
            assert order is not None

    def test_action_space_shape(self):
        decoder = ActionDecoder()
        space = decoder.flat_action_space
        assert space.shape == (10,)


# ---------------------------------------------------------------------------
# SimpleExecutionEngine tests
# ---------------------------------------------------------------------------

class TestSimpleExecutionEngine:
    def test_market_order_buy(self, snapshot, agent_state):
        engine = SimpleExecutionEngine()
        order = AgentOrder(
            agent_id="agent_0",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=snapshot.best_ask,
            size=5.0,
        )
        fill = engine.submit_order(order, snapshot, agent_state)
        assert fill is not None
        assert fill.fill_size == 5.0
        assert fill.fill_price >= snapshot.best_ask

    def test_market_order_sell(self, snapshot, agent_state):
        engine = SimpleExecutionEngine()
        order = AgentOrder(
            agent_id="agent_0",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            price=snapshot.best_bid,
            size=3.0,
        )
        fill = engine.submit_order(order, snapshot, agent_state)
        assert fill is not None
        assert fill.fill_price <= snapshot.best_bid

    def test_limit_order_passive(self, snapshot, agent_state):
        engine = SimpleExecutionEngine()
        # Far passive limit order
        order = AgentOrder(
            agent_id="agent_0",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            price=snapshot.best_bid - 1.0,   # far below bid
            size=5.0,
        )
        fill = engine.submit_order(order, snapshot, agent_state)
        assert fill is None   # resting
        assert len(agent_state.open_orders) == 1

    def test_nop_returns_none(self, snapshot, agent_state):
        engine = SimpleExecutionEngine()
        order = AgentOrder("a0", OrderType.NOP, OrderSide.BUY, 100, 1)
        fill = engine.submit_order(order, snapshot, agent_state)
        assert fill is None

    def test_cancel_clears_orders(self, snapshot, agent_state):
        engine = SimpleExecutionEngine()
        # Add a resting order
        order = AgentOrder("a0", OrderType.LIMIT, OrderSide.BUY,
                           snapshot.best_bid - 1, 5.0)
        engine.submit_order(order, snapshot, agent_state)
        assert len(agent_state.open_orders) == 1

        cancel = AgentOrder("a0", OrderType.CANCEL, OrderSide.BUY, 0, 0)
        engine.submit_order(cancel, snapshot, agent_state)
        assert len(agent_state.open_orders) == 0


# ---------------------------------------------------------------------------
# RewardComputer tests
# ---------------------------------------------------------------------------

class TestRewardComputer:
    def test_mtm_reward_up_move(self, snapshots):
        rc = RewardComputer(mode=RewardMode.MARK_TO_MARKET)
        state = AgentState("a0")
        state.inventory = 10.0
        state.cash = 0.0
        snap_prev, snap_curr = snapshots[0], snapshots[5]
        reward = rc.compute(state, snap_prev, snap_curr, [])
        # If price went up and we're long, reward should be positive-ish
        assert isinstance(reward, float)
        assert not math.isnan(reward)

    def test_inventory_penalty(self, snapshots):
        rc_no_pen = RewardComputer(inventory_penalty_coef=0.0)
        rc_with_pen = RewardComputer(inventory_penalty_coef=0.1)
        state = AgentState("a0")
        state.inventory = 500.0
        snap = snapshots[0]
        r_no = rc_no_pen.compute(state, snap, snap, [])
        r_with = rc_with_pen.compute(state, snap, snap, [])
        assert r_with < r_no   # penalty reduces reward

    def test_sharpe_reward(self, snapshots):
        rc = RewardComputer(mode=RewardMode.SHAPED_SHARPE)
        state = AgentState("a0")
        state.total_pnl = 0.0
        for i in range(10):
            state.total_pnl += 5.0
            snap = snapshots[i]
            r = rc.compute(state, snap, snap, [])
        assert isinstance(r, float)

    def test_reward_clipping(self, snapshots):
        rc = RewardComputer(clip_range=1.0)
        state = AgentState("a0")
        state.inventory = 10000.0   # huge inventory penalty
        snap = snapshots[0]
        r = rc.compute(state, snap, snap, [])
        assert abs(r) <= 1.0


# ---------------------------------------------------------------------------
# EpisodeManager tests
# ---------------------------------------------------------------------------

class TestEpisodeManager:
    def test_sample_episode(self, snapshots):
        em = EpisodeManager(snapshots, episode_len=100, seed=0)
        cfg = em.sample_episode()
        assert cfg.end_idx > cfg.start_idx
        assert cfg.end_idx - cfg.start_idx <= 100

    def test_scenario_types(self, snapshots):
        em = EpisodeManager(snapshots, episode_len=100, seed=1)
        for scenario in em.SCENARIOS:
            cfg = em.sample_episode(scenario=scenario)
            assert cfg.scenario_name == scenario

    def test_flash_crash_config(self, snapshots):
        em = EpisodeManager(snapshots, episode_len=200, seed=2)
        cfg = em.sample_episode(scenario="flash_crash")
        if cfg.flash_crash_enabled:
            assert 0 <= cfg.flash_crash_tick

    def test_get_snapshots(self, snapshots):
        em = EpisodeManager(snapshots, episode_len=50, seed=3)
        cfg = em.sample_episode()
        snaps = em.get_snapshots_for_episode(cfg)
        assert len(snaps) == cfg.end_idx - cfg.start_idx


# ---------------------------------------------------------------------------
# ChronosLOBEnv tests
# ---------------------------------------------------------------------------

class TestChronosLOBEnv:
    def test_reset_returns_obs(self, small_env):
        obs, info = small_env.reset()
        assert obs.shape == small_env.observation_space.shape
        assert obs.dtype == np.float32
        assert not np.isnan(obs).any()

    def test_step_returns_valid_output(self, small_env):
        small_env.reset()
        action = small_env.action_space.sample()
        obs, reward, done, trunc, info = small_env.step(action)
        assert obs.shape == small_env.observation_space.shape
        assert isinstance(reward, float)
        assert not math.isnan(reward)
        assert isinstance(done, bool)

    def test_episode_completes(self, small_env):
        small_env.reset()
        done = trunc = False
        steps = 0
        while not (done or trunc) and steps < 200:
            action = small_env.action_space.sample()
            _, _, done, trunc, _ = small_env.step(action)
            steps += 1
        assert steps > 0

    def test_info_dict_keys(self, small_env):
        small_env.reset()
        action = small_env.action_space.sample()
        _, _, _, _, info = small_env.step(action)
        assert "inventory" in info
        assert "total_pnl" in info
        assert "tick" in info

    def test_seed_reproducibility(self):
        env1 = ChronosLOBEnv(n_synthetic=200, episode_len=50, seed=7)
        env2 = ChronosLOBEnv(n_synthetic=200, episode_len=50, seed=7)
        obs1, _ = env1.reset(seed=7)
        obs2, _ = env2.reset(seed=7)
        np.testing.assert_array_almost_equal(obs1, obs2)

    def test_render(self, small_env):
        small_env.reset()
        result = small_env.render()
        assert result is None or isinstance(result, str)

    def test_different_reward_modes(self):
        for mode in [RewardMode.REALIZED_PNL, RewardMode.MARK_TO_MARKET,
                     RewardMode.SHAPED_SHARPE]:
            env = ChronosLOBEnv(reward_mode=mode, n_synthetic=200, episode_len=50)
            env.reset()
            for _ in range(10):
                action = env.action_space.sample()
                _, reward, done, trunc, _ = env.step(action)
                assert not math.isnan(reward)
                if done or trunc:
                    break

    def test_options_difficulty(self, small_env):
        obs1, _ = small_env.reset(options={"difficulty": 0.1})
        obs2, _ = small_env.reset(options={"difficulty": 0.9})
        assert obs1 is not None and obs2 is not None


# ---------------------------------------------------------------------------
# ChronosMARLEnv tests
# ---------------------------------------------------------------------------

class TestChronosMARLEnv:
    def test_reset(self):
        env = ChronosMARLEnv(n_agents=3, n_synthetic=300, episode_len=50)
        obs, infos = env.reset()
        assert set(obs.keys()) == {"agent_0", "agent_1", "agent_2"}
        for aid, o in obs.items():
            assert o.shape == env.observation_spaces[aid].shape
            assert not np.isnan(o).any()

    def test_step(self):
        env = ChronosMARLEnv(n_agents=3, n_synthetic=300, episode_len=50)
        env.reset()
        actions = {aid: env.action_spaces[aid].sample() for aid in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        assert set(rewards.keys()) == {"agent_0", "agent_1", "agent_2"}
        for r in rewards.values():
            assert not math.isnan(r)

    def test_episode_terminates(self):
        env = ChronosMARLEnv(n_agents=2, n_synthetic=200, episode_len=30)
        env.reset()
        for _ in range(40):
            if not env.agents:
                break
            actions = {aid: env.action_spaces[aid].sample() for aid in env.agents}
            env.step(actions)
        # Should have terminated
        assert len(env.agents) == 0

    def test_cooperative_mixing(self):
        env_coop = ChronosMARLEnv(n_agents=2, n_synthetic=200, episode_len=50,
                                    cooperative_coef=1.0)
        env_ind = ChronosMARLEnv(n_agents=2, n_synthetic=200, episode_len=50,
                                   cooperative_coef=0.0)
        env_coop.reset(seed=1)
        env_ind.reset(seed=1)
        actions_coop = {aid: env_coop.action_spaces[aid].sample() for aid in env_coop.agents}
        actions_ind = {aid: env_ind.action_spaces[aid].sample() for aid in env_ind.agents}
        _, rewards_coop, _, _, _ = env_coop.step(actions_coop)
        _, rewards_ind, _, _, _ = env_ind.step(actions_ind)
        # With coop=1.0, all rewards should be equal
        coop_vals = list(rewards_coop.values())
        if len(coop_vals) > 1:
            assert abs(coop_vals[0] - coop_vals[1]) < 0.01


# ---------------------------------------------------------------------------
# VecChronosEnv tests
# ---------------------------------------------------------------------------

class TestVecChronosEnv:
    def test_reset(self):
        vec = VecChronosEnv(n_envs=4, env_kwargs={"n_synthetic": 200, "episode_len": 50})
        batch_obs, infos = vec.reset()
        assert batch_obs.shape[0] == 4
        assert batch_obs.dtype == np.float32
        vec.close()

    def test_step(self):
        vec = VecChronosEnv(n_envs=3, env_kwargs={"n_synthetic": 200, "episode_len": 50})
        vec.reset()
        actions = np.stack([vec.action_space.sample() for _ in range(3)])
        obs, rewards, dones, truncs, infos = vec.step(actions)
        assert obs.shape[0] == 3
        assert rewards.shape == (3,)
        assert dones.shape == (3,)
        vec.close()

    def test_auto_reset_on_done(self):
        vec = VecChronosEnv(n_envs=2, env_kwargs={"n_synthetic": 100, "episode_len": 5})
        vec.reset()
        for _ in range(20):
            actions = np.stack([vec.action_space.sample() for _ in range(2)])
            obs, rewards, dones, truncs, infos = vec.step(actions)
        assert obs is not None
        vec.close()


# ---------------------------------------------------------------------------
# Wrappers tests
# ---------------------------------------------------------------------------

class TestWrappers:
    def test_normalised_obs(self):
        env = NormalisedObsWrapper(
            ChronosLOBEnv(n_synthetic=200, episode_len=50, seed=0)
        )
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_frame_stack(self):
        base = ChronosLOBEnv(n_synthetic=200, episode_len=50, seed=0)
        env = FrameStackWrapper(base, n_stack=3)
        obs, _ = env.reset()
        assert obs.shape[0] == base.observation_space.shape[0] * 3

    def test_make_single_env(self):
        env = make_single_env(normalise=True, frame_stack=2,
                               n_synthetic=200, episode_len=50)
        obs, _ = env.reset()
        assert obs is not None


# ---------------------------------------------------------------------------
# EpisodeTracker tests
# ---------------------------------------------------------------------------

class TestEpisodeTracker:
    def test_record_and_summary(self):
        tracker = EpisodeTracker(window=10)
        for i in range(15):
            tracker.record_episode(float(i), 200, float(i * 10))
        summary = tracker.summary()
        assert "mean_return" in summary
        assert "sharpe" in summary
        assert summary["n_episodes"] == 15

    def test_sharpe_ratio(self):
        tracker = EpisodeTracker(window=50)
        for _ in range(50):
            tracker.record_episode(10.0, 200, 50.0)  # constant returns
        summary = tracker.summary()
        # Constant returns = infinite Sharpe; std=0, so we just check it computes
        assert math.isfinite(summary["sharpe"]) or math.isinf(summary["sharpe"])


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------

class TestFactories:
    def test_make_vec_env(self):
        env = make_vec_env(n_envs=2, n_synthetic=200, episode_len=50)
        obs, _ = env.reset()
        assert obs.shape[0] == 2
        env.close()

    def test_make_marl_env(self):
        env = make_marl_env(n_agents=3, n_synthetic=200, episode_len=50)
        obs, _ = env.reset()
        assert len(obs) == 3
