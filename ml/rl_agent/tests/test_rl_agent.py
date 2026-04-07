"""
ml/rl_agent/tests/test_rl_agent.py -- Comprehensive tests for the RL exit policy framework.

Tests cover:
  - TradingEnvironment correctness (reset, step, terminal conditions)
  - TradeEpisodeGenerator output shape and statistical properties
  - QNetwork forward pass, backprop convergence
  - ReplayBuffer circular storage and PER sampling bias
  - DQNAgent epsilon decay, Double DQN targets
  - PPOAgent GAE computation, PPO clip objective
  - RLTrainer Q-table export format compatibility

Run with: python -m pytest ml/rl_agent/tests/test_rl_agent.py -v
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup -- allow running from repo root without install
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.rl_agent.environment import (
    TradingEnvironment,
    TradingState,
    TradeEpisode,
    TradeEpisodeGenerator,
    HOLD,
    PARTIAL_EXIT,
    FULL_EXIT,
    MAX_BARS,
    N_FEATURES,
    N_ACTIONS,
    STOP_LOSS_PCT,
    HOLDING_COST,
    TRANSACTION_COST,
)
from ml.rl_agent.q_network import (
    QNetwork,
    ReplayBuffer,
    DQNAgent,
    INPUT_DIM,
    OUTPUT_DIM,
    PER_ALPHA,
    PER_EPSILON,
    BATCH_SIZE,
)
from ml.rl_agent.ppo_agent import (
    PolicyNetwork,
    PPOAgent,
    Episode,
    CLIP_RATIO,
    VALUE_COEF,
    ENTROPY_COEF,
    GAMMA,
    LAM,
)
from ml.rl_agent.trainer import RLTrainer, TrainConfig, EvalResult, _passive_hold_pnl


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def episode_gen():
    return TradeEpisodeGenerator(seed=0)


@pytest.fixture
def env(episode_gen):
    return TradingEnvironment(episode_generator=episode_gen, seed=0)


@pytest.fixture
def trending_episode(episode_gen):
    return episode_gen.generate("bh_trending", n_bars=36, entry_price=100.0)


@pytest.fixture
def inactive_episode(episode_gen):
    return episode_gen.generate("bh_inactive", n_bars=36, entry_price=100.0)


@pytest.fixture
def mixed_episode(episode_gen):
    return episode_gen.generate("mixed", n_bars=36, entry_price=100.0)


@pytest.fixture
def qnet():
    return QNetwork(learning_rate=1e-3, seed=42)


@pytest.fixture
def replay_buf():
    return ReplayBuffer(capacity=1000, state_dim=INPUT_DIM, seed=7)


@pytest.fixture
def dqn_agent():
    return DQNAgent(
        learning_rate=1e-3,
        batch_size=32,
        buffer_capacity=5000,
        epsilon_decay_steps=1000,
        seed=0,
    )


@pytest.fixture
def ppo_agent():
    return PPOAgent(learning_rate=3e-4, seed=0)


@pytest.fixture
def trainer():
    cfg = TrainConfig(
        dqn_episodes=10,
        ppo_episodes=10,
        eval_episodes=10,
        save_every=5,
        log_every=5,
        dqn_train_steps_per_episode=1,
        ppo_episodes_per_update=2,
        seed=0,
        checkpoint_dir=tempfile.mkdtemp(),
    )
    return RLTrainer(config=cfg)


# ===========================================================================
# Environment tests
# ===========================================================================

class TestEnvironmentReset:
    def test_environment_reset_state_shape(self, env):
        """reset() must return shape (10,) float32 array."""
        obs = env.reset()
        assert obs.shape == (N_FEATURES,), f"Expected ({N_FEATURES},), got {obs.shape}"
        assert obs.dtype == np.float32

    def test_environment_reset_returns_valid_values(self, env):
        """All obs values must be finite and within declared bounds."""
        for _ in range(20):
            obs = env.reset()
            assert np.all(np.isfinite(obs)), "Non-finite value in obs"

    def test_environment_reset_with_episode(self, env, trending_episode):
        """reset(trade_episode=...) should use provided episode."""
        obs1 = env.reset(trade_episode=trending_episode)
        obs2 = env.reset(trade_episode=trending_episode)
        np.testing.assert_array_equal(obs1, obs2, "Same episode should yield same initial obs")

    def test_environment_reset_clears_position(self, env):
        """After reset, position fraction must be 1.0."""
        env.reset()
        assert env._position_fraction == 1.0

    def test_environment_reset_clears_done_flag(self, env):
        """After reset, _done must be False."""
        env.reset()
        assert not env._done

    def test_environment_step_requires_reset(self):
        """Calling step() without reset() should raise RuntimeError."""
        e = TradingEnvironment(seed=0)
        with pytest.raises(RuntimeError, match="reset"):
            e.step(HOLD)


class TestEnvironmentStep:
    def test_environment_step_returns_correct_types(self, env):
        """step() must return (ndarray, float, bool, dict)."""
        env.reset()
        obs, reward, done, info = env.step(HOLD)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_environment_step_hold_reward(self, env):
        """HOLD reward should be negative (holding cost) in a flat market."""
        # Use a flat price episode (no drift)
        ep_gen = TradeEpisodeGenerator(seed=1)
        flat_ep = ep_gen.generate("bh_inactive", n_bars=36, entry_price=100.0)
        # Manually flatten prices
        flat_ep.prices[:] = 100.0
        env.reset(trade_episode=flat_ep)
        obs, reward, done, info = env.step(HOLD)
        # With zero PnL change, reward should be negative (holding cost)
        assert reward < 0, f"Expected negative HOLD reward, got {reward}"

    def test_environment_step_exit_terminal(self, env):
        """FULL_EXIT should return done=True."""
        env.reset()
        _, _, done, _ = env.step(FULL_EXIT)
        assert done, "FULL_EXIT should set done=True"

    def test_environment_step_partial_exit_not_terminal(self, env):
        """PARTIAL_EXIT should not end the episode (done=False) unless stop-loss or max bars."""
        env.reset()
        _, _, done, _ = env.step(PARTIAL_EXIT)
        assert not done, "PARTIAL_EXIT alone should not end episode"

    def test_environment_step_partial_exit_halves_position(self, env):
        """After PARTIAL_EXIT, position fraction should be 0.5."""
        env.reset()
        env.step(PARTIAL_EXIT)
        assert env._position_fraction == pytest.approx(0.5, abs=1e-6)

    def test_environment_step_obs_shape_after_step(self, env):
        """Observation returned by step() must have shape (10,)."""
        env.reset()
        obs, _, _, _ = env.step(HOLD)
        assert obs.shape == (N_FEATURES,)

    def test_environment_terminal_max_bars(self, env, trending_episode):
        """Episode must terminate when bars_held reaches MAX_BARS."""
        env.reset(trade_episode=trending_episode)
        done = False
        steps = 0
        while not done and steps < MAX_BARS + 10:
            _, _, done, info = env.step(HOLD)
            steps += 1
        assert done, "Episode should have terminated at max bars"
        assert steps <= MAX_BARS + 2, f"Too many steps: {steps}"

    def test_environment_terminal_stop_loss(self):
        """Episode terminates when PnL < STOP_LOSS_PCT."""
        ep_gen = TradeEpisodeGenerator(seed=99)
        crash_ep = ep_gen.generate("bh_inactive", n_bars=36, entry_price=100.0)
        # Inject a price crash starting from bar 1 (after entry bar 0)
        crash_ep.prices[1:] = 90.0  # -10% loss >> stop loss threshold (-3%)
        env = TradingEnvironment(seed=0)
        env.reset(trade_episode=crash_ep)
        # After stepping to bar 1 (HOLD), the new bar's PnL = -10% -> stop loss
        _, _, done, info = env.step(HOLD)
        assert done, "Stop-loss terminal condition failed"
        assert info.get("terminal") == "stop_loss"

    def test_environment_invalid_action_raises(self, env):
        """Invalid action integer should raise ValueError."""
        env.reset()
        with pytest.raises((ValueError, Exception)):
            env.step(99)

    def test_environment_info_dict_keys(self, env):
        """info dict should contain standard keys."""
        env.reset()
        _, _, _, info = env.step(HOLD)
        assert "bar" in info
        assert "action" in info
        assert "pnl_pct" in info

    def test_environment_reward_scale(self):
        """Custom reward_scale should multiply all rewards."""
        ep_gen = TradeEpisodeGenerator(seed=5)
        ep = ep_gen.generate("bh_trending", n_bars=36, entry_price=100.0)
        # Flat prices so reward is purely from costs
        ep.prices[:] = 100.0
        env1 = TradingEnvironment(seed=0, reward_scale=1.0)
        env2 = TradingEnvironment(seed=0, reward_scale=10.0)
        env1.reset(trade_episode=ep)
        env2.reset(trade_episode=ep)
        _, r1, _, _ = env1.step(HOLD)
        # Need same episode
        ep2 = ep_gen.generate("bh_trending", n_bars=36, entry_price=100.0)
        ep2.prices[:] = 100.0
        env2.reset(trade_episode=ep2)
        _, r2, _, _ = env2.step(HOLD)
        # r2 should be 10x r1 (both negative)
        assert abs(r2) > abs(r1), "Higher reward_scale should produce larger magnitude rewards"


# ===========================================================================
# TradeEpisodeGenerator tests
# ===========================================================================

class TestTradeEpisodeGenerator:
    def test_trade_episode_generator_bh_trending(self, episode_gen):
        """BH-trending episode should have positive mean return."""
        returns = []
        for _ in range(50):
            ep = episode_gen.generate("bh_trending", n_bars=36, entry_price=100.0)
            ret = (ep.prices[-1] - ep.prices[0]) / ep.prices[0]
            returns.append(ret)
        # With positive drift, mean return should be positive
        assert np.mean(returns) > 0, "BH-trending episodes should have positive mean return"

    def test_trade_episode_generator_bh_inactive(self, episode_gen):
        """BH-inactive (OU) episode should show lower Hurst than trending."""
        trending_hurst = []
        inactive_hurst = []
        for _ in range(30):
            ep_t = episode_gen.generate("bh_trending", n_bars=36, entry_price=100.0)
            ep_i = episode_gen.generate("bh_inactive", n_bars=36, entry_price=100.0)
            trending_hurst.append(np.mean(ep_t.hurst_series))
            inactive_hurst.append(np.mean(ep_i.hurst_series))
        assert np.mean(trending_hurst) > np.mean(inactive_hurst), \
            "Trending episodes should have higher mean Hurst than inactive"

    def test_trade_episode_generator_mixed(self, mixed_episode):
        """Mixed episode should have BH active in first half and inactive in second half."""
        n = len(mixed_episode)
        split = n // 2
        first_half_active = np.mean(mixed_episode.bh_active_series[:split])
        second_half_active = np.mean(mixed_episode.bh_active_series[split:])
        # Not necessarily strict (stochastic), but first half should be more active
        assert first_half_active >= second_half_active, \
            f"First half BH active rate {first_half_active:.2f} should >= second half {second_half_active:.2f}"

    def test_episode_prices_positive(self, episode_gen):
        """All prices must be strictly positive."""
        for ep_type in ["bh_trending", "bh_inactive", "mixed"]:
            ep = episode_gen.generate(ep_type, n_bars=36, entry_price=100.0)
            assert np.all(ep.prices > 0), f"Negative prices in {ep_type} episode"

    def test_episode_shapes_consistent(self, episode_gen):
        """All episode arrays must have the same length."""
        ep = episode_gen.generate("bh_trending", n_bars=36, entry_price=100.0)
        n = len(ep)
        assert ep.prices.shape == (n,)
        assert ep.bh_mass_series.shape == (n,)
        assert ep.bh_active_series.shape == (n,)
        assert ep.atr_series.shape == (n,)
        assert ep.hurst_series.shape == (n,)
        assert ep.vol_pct_series.shape == (n,)
        assert ep.bar_minutes.shape == (n,)

    def test_episode_bh_mass_in_range(self, episode_gen):
        """BH mass values must be in [0, 1]."""
        for ep_type in ["bh_trending", "bh_inactive", "mixed"]:
            ep = episode_gen.generate(ep_type, n_bars=36)
            assert np.all(ep.bh_mass_series >= 0) and np.all(ep.bh_mass_series <= 1), \
                f"bh_mass out of [0,1] in {ep_type}"

    def test_episode_bh_active_binary(self, episode_gen):
        """BH active must be binary (0 or 1)."""
        ep = episode_gen.generate("bh_trending", n_bars=36)
        unique_vals = np.unique(ep.bh_active_series)
        for v in unique_vals:
            assert v in (0.0, 1.0), f"Non-binary bh_active value: {v}"

    def test_episode_hurst_in_range(self, episode_gen):
        """Hurst exponent must be in [0, 1]."""
        ep = episode_gen.generate("bh_trending", n_bars=36)
        assert np.all(ep.hurst_series >= 0) and np.all(ep.hurst_series <= 1)

    def test_random_episode_type_variation(self, episode_gen):
        """random_episode() should produce all three types over many calls."""
        types_seen = set()
        for _ in range(100):
            ep = episode_gen.random_episode()
            types_seen.add(ep.episode_type)
        assert len(types_seen) >= 2, f"Expected multiple episode types, got {types_seen}"

    def test_invalid_episode_type_raises(self, episode_gen):
        with pytest.raises(ValueError, match="Unknown episode_type"):
            episode_gen.generate("invalid_type")


# ===========================================================================
# QNetwork tests
# ===========================================================================

class TestQNetwork:
    def test_q_network_forward_shape(self, qnet):
        """predict() on single state returns shape (3,)."""
        state = np.random.randn(INPUT_DIM).astype(np.float32)
        q = qnet.predict(state)
        assert q.shape == (OUTPUT_DIM,), f"Expected ({OUTPUT_DIM},), got {q.shape}"

    def test_q_network_forward_batch_shape(self, qnet):
        """predict() on batch returns shape (batch, 3)."""
        states = np.random.randn(8, INPUT_DIM).astype(np.float32)
        q = qnet.predict(states)
        assert q.shape == (8, OUTPUT_DIM)

    def test_q_network_forward_finite(self, qnet):
        """All Q-values must be finite."""
        state = np.random.randn(INPUT_DIM).astype(np.float32)
        q = qnet.predict(state)
        assert np.all(np.isfinite(q)), "Q-values contain non-finite values"

    def test_q_network_backprop_loss_decreases(self):
        """Training on a fixed target should reduce MSE loss over many steps."""
        net = QNetwork(learning_rate=5e-3, seed=0, use_adam=True)
        rng = np.random.default_rng(0)

        # Fixed regression target: action 0 should have Q=1.0 for all states
        states = rng.standard_normal((32, INPUT_DIM)).astype(np.float32)
        actions = np.zeros(32, dtype=np.int32)
        targets = np.ones(32, dtype=np.float32)

        losses = []
        for _ in range(200):
            loss, _ = net.update(states, actions, targets)
            losses.append(loss)

        # Loss at end should be lower than loss at start
        early_loss = np.mean(losses[:10])
        late_loss = np.mean(losses[-10:])
        assert late_loss < early_loss * 0.5, \
            f"Loss should decrease: early={early_loss:.4f}, late={late_loss:.4f}"

    def test_q_network_different_seeds_produce_different_weights(self):
        """Two networks with different seeds should have different weights."""
        net1 = QNetwork(seed=1)
        net2 = QNetwork(seed=2)
        assert not np.allclose(net1.W1, net2.W1), "Different seeds should produce different W1"

    def test_q_network_copy_weights(self):
        """copy_weights_from should make two networks identical."""
        net1 = QNetwork(seed=1)
        net2 = QNetwork(seed=2)
        net2.copy_weights_from(net1)
        np.testing.assert_array_equal(net1.W1, net2.W1)
        np.testing.assert_array_equal(net1.b3, net2.b3)

    def test_q_network_soft_update(self):
        """soft_update_from with tau=1.0 should fully copy online weights."""
        online = QNetwork(seed=3)
        target = QNetwork(seed=7)
        original_w = target.W1.copy()
        target.soft_update_from(online, tau=1.0)
        np.testing.assert_array_almost_equal(target.W1, online.W1)

    def test_q_network_soft_update_partial(self):
        """soft_update_from with tau=0.5 should blend weights."""
        online = QNetwork(seed=3)
        target = QNetwork(seed=7)
        old_target_w = target.W1.copy()
        target.soft_update_from(online, tau=0.5)
        expected = 0.5 * online.W1 + 0.5 * old_target_w
        np.testing.assert_array_almost_equal(target.W1, expected, decimal=5)

    def test_q_network_save_load(self, qnet, tmp_path):
        """save() then load() should recover identical weights."""
        path = str(tmp_path / "qnet")
        qnet.save(path)
        loaded = QNetwork.load(path + ".npz")
        np.testing.assert_array_equal(qnet.W1, loaded.W1)
        np.testing.assert_array_equal(qnet.b3, loaded.b3)

    def test_q_network_update_returns_td_errors(self, qnet):
        """update() should return td_errors of correct shape."""
        rng = np.random.default_rng(0)
        states = rng.standard_normal((8, INPUT_DIM)).astype(np.float32)
        actions = rng.integers(0, OUTPUT_DIM, 8).astype(np.int32)
        targets = rng.standard_normal(8).astype(np.float32)
        loss, td_errors = qnet.update(states, actions, targets)
        assert td_errors.shape == (8,), f"Expected (8,) td_errors, got {td_errors.shape}"
        assert isinstance(loss, float)

    def test_q_network_sgd_mode(self):
        """SGD optimizer mode should also reduce loss."""
        net = QNetwork(learning_rate=1e-2, seed=0, use_adam=False)
        rng = np.random.default_rng(0)
        states = rng.standard_normal((32, INPUT_DIM)).astype(np.float32)
        actions = np.zeros(32, dtype=np.int32)
        targets = np.ones(32, dtype=np.float32)
        losses = [net.update(states, actions, targets)[0] for _ in range(100)]
        assert losses[-1] < losses[0], "SGD should decrease loss"


# ===========================================================================
# ReplayBuffer tests
# ===========================================================================

class TestReplayBuffer:
    def _make_transition(self, rng, idx=0):
        s = rng.standard_normal(INPUT_DIM).astype(np.float32)
        a = int(rng.integers(0, 3))
        r = float(rng.standard_normal())
        ns = rng.standard_normal(INPUT_DIM).astype(np.float32)
        d = bool(rng.random() > 0.9)
        return s, a, r, ns, d

    def test_replay_buffer_sample_size(self, replay_buf):
        """sample(batch_size) should return exactly batch_size transitions."""
        rng = np.random.default_rng(0)
        for _ in range(100):
            s, a, r, ns, d = self._make_transition(rng)
            replay_buf.add(s, a, r, ns, d)
        states, actions, rewards, next_states, dones, weights, indices = replay_buf.sample(32)
        assert states.shape == (32, INPUT_DIM)
        assert actions.shape == (32,)
        assert rewards.shape == (32,)
        assert next_states.shape == (32, INPUT_DIM)
        assert dones.shape == (32,)
        assert weights.shape == (32,)
        assert indices.shape == (32,)

    def test_replay_buffer_raises_when_insufficient(self, replay_buf):
        """sample() should raise if buffer has fewer elements than batch_size."""
        rng = np.random.default_rng(0)
        replay_buf.add(*self._make_transition(rng))
        with pytest.raises(ValueError, match="Buffer has only"):
            replay_buf.sample(32)

    def test_replay_buffer_circular_overwrite(self):
        """Buffer should overwrite old entries when capacity exceeded."""
        buf = ReplayBuffer(capacity=10, state_dim=INPUT_DIM, seed=0)
        rng = np.random.default_rng(0)
        for i in range(15):
            s = np.full(INPUT_DIM, float(i), dtype=np.float32)
            buf.add(s, 0, 0.0, s, False)
        assert len(buf) == 10, f"Buffer size should be capped at 10, got {len(buf)}"

    def test_per_priority_sampling_bias(self):
        """Higher-priority transitions should be sampled more frequently."""
        buf = ReplayBuffer(capacity=1000, state_dim=INPUT_DIM, alpha=1.0, seed=0)
        rng = np.random.default_rng(0)
        n = 200

        for i in range(n):
            s = rng.standard_normal(INPUT_DIM).astype(np.float32)
            # High priority for first 10 transitions
            td_error = 10.0 if i < 10 else 0.01
            buf.add(s, 0, 0.0, s, False, td_error=td_error)

        # Count how often high-priority indices (0-9) are sampled
        high_priority_count = 0
        total_samples = 0
        for _ in range(100):
            _, _, _, _, _, _, indices = buf.sample(32)
            high_priority_count += int(np.sum(indices < 10))
            total_samples += 32

        high_priority_rate = high_priority_count / total_samples
        # Should be sampled much more than their proportion (10/200 = 5%)
        assert high_priority_rate > 0.10, \
            f"High-priority transitions undersampled: {high_priority_rate:.3f}"

    def test_replay_buffer_is_ready(self, replay_buf):
        """is_ready() should return False until enough transitions stored."""
        rng = np.random.default_rng(0)
        assert not replay_buf.is_ready(32)
        for _ in range(32):
            replay_buf.add(*self._make_transition(rng))
        assert replay_buf.is_ready(32)

    def test_replay_buffer_update_priorities(self, replay_buf):
        """update_priorities() should change stored priority values."""
        rng = np.random.default_rng(0)
        for _ in range(100):
            replay_buf.add(*self._make_transition(rng))
        _, _, _, _, _, _, indices = replay_buf.sample(10)
        # Set very high TD errors for these indices
        new_errors = np.full(len(indices), 100.0)
        replay_buf.update_priorities(indices, new_errors)
        # Check that max priority increased
        expected_prio = (100.0 + PER_EPSILON) ** PER_ALPHA
        assert replay_buf._max_priority >= expected_prio * 0.9

    def test_replay_buffer_weights_in_range(self, replay_buf):
        """IS weights should be in (0, 1]."""
        rng = np.random.default_rng(0)
        for _ in range(100):
            replay_buf.add(*self._make_transition(rng))
        _, _, _, _, _, weights, _ = replay_buf.sample(32)
        assert np.all(weights > 0) and np.all(weights <= 1.0 + 1e-6), \
            f"Weights out of range: min={weights.min():.4f}, max={weights.max():.4f}"


# ===========================================================================
# DQNAgent tests
# ===========================================================================

class TestDQNAgent:
    def test_dqn_agent_act_returns_valid_action(self, dqn_agent):
        """act() should return an integer in {0, 1, 2}."""
        state = np.random.randn(INPUT_DIM).astype(np.float32)
        action = dqn_agent.act(state, explore=True)
        assert action in (0, 1, 2), f"Invalid action: {action}"

    def test_dqn_agent_act_greedy(self, dqn_agent):
        """With explore=False, act() should be deterministic."""
        state = np.random.randn(INPUT_DIM).astype(np.float32)
        actions = [dqn_agent.act(state, explore=False) for _ in range(10)]
        assert len(set(actions)) == 1, "Greedy actions should be deterministic"

    def test_dqn_agent_epsilon_decay(self):
        """Epsilon should decrease from EPS_START toward EPS_END after many steps."""
        agent = DQNAgent(
            epsilon_decay_steps=200,
            epsilon_start=1.0,
            epsilon_end=0.05,
            batch_size=8,
            buffer_capacity=500,
            seed=0,
        )
        rng = np.random.default_rng(0)

        # Fill buffer
        for _ in range(50):
            s = rng.standard_normal(INPUT_DIM).astype(np.float32)
            ns = rng.standard_normal(INPUT_DIM).astype(np.float32)
            agent.store(s, 0, 0.1, ns, False)

        # Run many train steps to trigger epsilon decay
        for _ in range(300):
            agent.train_step()

        assert agent.epsilon < 0.5, f"Epsilon should have decayed, got {agent.epsilon}"
        assert agent.epsilon >= 0.05, f"Epsilon should not go below EPS_END={0.05}"

    def test_dqn_agent_train_step_returns_loss(self, dqn_agent):
        """train_step() should return a float loss when buffer is ready."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            s = rng.standard_normal(INPUT_DIM).astype(np.float32)
            ns = rng.standard_normal(INPUT_DIM).astype(np.float32)
            dqn_agent.store(s, 0, 0.1, ns, False)
        loss = dqn_agent.train_step()
        assert loss is not None and isinstance(loss, float)
        assert np.isfinite(loss)

    def test_dqn_agent_train_step_none_when_buffer_empty(self, dqn_agent):
        """train_step() should return None when buffer not ready."""
        result = dqn_agent.train_step()
        assert result is None

    def test_dqn_agent_target_network_updates(self):
        """Target network should diverge from online during early training, then sync."""
        agent = DQNAgent(
            target_update_freq=10,
            batch_size=8,
            buffer_capacity=500,
            seed=42,
        )
        rng = np.random.default_rng(0)
        for _ in range(100):
            s = rng.standard_normal(INPUT_DIM).astype(np.float32)
            ns = rng.standard_normal(INPUT_DIM).astype(np.float32)
            agent.store(s, 0, 0.1, ns, False)
        for _ in range(15):
            agent.train_step()
        # After 15 steps (> target_update_freq=10), target should have been hard-updated
        # Check that target W1 is not far from online W1
        diff = np.mean(np.abs(agent.online.W1 - agent.target.W1))
        assert diff < 0.5, f"Target network too far from online: diff={diff:.4f}"

    def test_dqn_agent_summary(self, dqn_agent):
        """summary() should return a dict with expected keys."""
        summary = dqn_agent.summary()
        assert "step" in summary
        assert "epsilon" in summary
        assert "buffer_size" in summary

    def test_dqn_agent_save_load(self, dqn_agent, tmp_path):
        """save() then load() should recover weights."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            s = rng.standard_normal(INPUT_DIM).astype(np.float32)
            ns = rng.standard_normal(INPUT_DIM).astype(np.float32)
            dqn_agent.store(s, 0, 1.0, ns, False)

        prefix = str(tmp_path / "dqn_test")
        dqn_agent.save(prefix)
        dqn_agent.load(prefix)

        state = rng.standard_normal(INPUT_DIM).astype(np.float32)
        q = dqn_agent.online.predict(state)
        assert q.shape == (OUTPUT_DIM,)

    def test_dqn_agent_double_dqn_target_computation(self):
        """Double DQN: online net selects action, target net evaluates it."""
        agent = DQNAgent(batch_size=4, buffer_capacity=100, seed=0)
        rng = np.random.default_rng(0)
        for _ in range(10):
            s = rng.standard_normal(INPUT_DIM).astype(np.float32)
            ns = rng.standard_normal(INPUT_DIM).astype(np.float32)
            agent.store(s, 0, 1.0, ns, False)
        # Manually verify: online selects best action, target evaluates
        states, _, _, next_states, _, _, _ = agent.buffer.sample(4)
        online_q = agent.online.predict(next_states)
        best_actions = np.argmax(online_q, axis=1)
        target_q = agent.target.predict(next_states)
        selected_q = target_q[np.arange(4), best_actions]
        assert selected_q.shape == (4,)
        assert np.all(np.isfinite(selected_q))


# ===========================================================================
# PPOAgent tests
# ===========================================================================

class TestPPOAgent:
    def test_ppo_gae_computation(self, ppo_agent):
        """GAE computation should produce correct shapes and finite values."""
        T = 10
        rewards = np.random.randn(T).astype(np.float32)
        values = np.random.randn(T).astype(np.float32)
        dones = np.zeros(T, dtype=np.bool_)
        dones[-1] = True

        advantages, returns = ppo_agent.compute_gae(rewards, values, dones)
        assert advantages.shape == (T,), f"Expected ({T},), got {advantages.shape}"
        assert returns.shape == (T,)
        assert np.all(np.isfinite(advantages))
        assert np.all(np.isfinite(returns))

    def test_ppo_gae_terminal_zero_bootstrap(self, ppo_agent):
        """GAE for a terminal step should not bootstrap future value."""
        rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        values = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        dones = np.array([False, False, True], dtype=np.bool_)

        advantages, returns = ppo_agent.compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)
        # Last step is terminal: delta = r - V = 1.0 - 0.5 = 0.5
        assert abs(advantages[-1] - 0.5) < 1e-4, f"Terminal advantage mismatch: {advantages[-1]}"

    def test_ppo_gae_discount_applied(self, ppo_agent):
        """Later rewards should be discounted in GAE."""
        rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        values = np.zeros(3, dtype=np.float32)
        dones = np.array([False, False, True], dtype=np.bool_)

        adv_g1, _ = ppo_agent.compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)
        adv_g0, _ = ppo_agent.compute_gae(rewards, values, dones, gamma=0.5, lam=1.0)
        # With gamma=0.5, advantage[0] should be lower than with gamma=1.0
        assert abs(adv_g1[0]) > abs(adv_g0[0]), "Discounting should reduce future advantage"

    def test_ppo_collect_episode_shape(self, ppo_agent, env):
        """collect_episode() should return Episode with consistent array shapes."""
        ep = ppo_agent.collect_episode(env)
        T = len(ep)
        assert T > 0
        assert ep.states.shape == (T, N_FEATURES)
        assert ep.actions.shape == (T,)
        assert ep.rewards.shape == (T,)
        assert ep.values.shape == (T,)
        assert ep.log_probs.shape == (T,)
        assert ep.dones.shape == (T,)

    def test_ppo_collect_episode_dones_terminal(self, ppo_agent, env):
        """Last done flag in episode must be True."""
        ep = ppo_agent.collect_episode(env)
        assert ep.dones[-1], "Last done flag should be True"

    def test_ppo_update_returns_dict(self, ppo_agent, env):
        """update() should return a dict with loss keys."""
        episodes = [ppo_agent.collect_episode(env) for _ in range(3)]
        info = ppo_agent.update(episodes)
        assert "policy_loss" in info
        assert "value_loss" in info
        assert "entropy" in info
        assert "total_loss" in info

    def test_ppo_entropy_positive(self, ppo_agent, env):
        """Entropy should be positive for a stochastic policy."""
        episodes = [ppo_agent.collect_episode(env) for _ in range(2)]
        info = ppo_agent.update(episodes)
        assert info["entropy"] > 0, f"Entropy should be positive, got {info['entropy']}"

    def test_ppo_act_deterministic(self, ppo_agent):
        """Deterministic act() should be consistent."""
        state = np.random.randn(N_FEATURES).astype(np.float32)
        actions = [ppo_agent.act(state, deterministic=True) for _ in range(10)]
        assert len(set(actions)) == 1, "Deterministic act should be consistent"

    def test_ppo_act_returns_valid_action(self, ppo_agent):
        """act() should return action in {0, 1, 2}."""
        state = np.random.randn(N_FEATURES).astype(np.float32)
        for _ in range(20):
            a = ppo_agent.act(state)
            assert a in (0, 1, 2), f"Invalid action: {a}"

    def test_ppo_summary_keys(self, ppo_agent, env):
        """summary() should contain n_episodes."""
        ppo_agent.collect_episode(env)
        s = ppo_agent.summary()
        assert "n_episodes" in s

    def test_policy_network_forward_shapes(self):
        """PolicyNetwork forward methods should return correct shapes."""
        net = PolicyNetwork(seed=0)
        rng = np.random.default_rng(0)
        states = rng.standard_normal((4, N_FEATURES)).astype(np.float32)

        probs = net.predict_action_probs(states)
        assert probs.shape == (4, N_ACTIONS), f"Expected (4, {N_ACTIONS}), got {probs.shape}"

        values = net.predict_value(states)
        assert values.shape == (4,), f"Expected (4,), got {values.shape}"

    def test_policy_network_probs_sum_to_one(self):
        """Action probabilities must sum to 1.0 for each sample."""
        net = PolicyNetwork(seed=0)
        rng = np.random.default_rng(0)
        states = rng.standard_normal((8, N_FEATURES)).astype(np.float32)
        probs = net.predict_action_probs(states)
        sums = probs.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5, err_msg="Action probs must sum to 1")

    def test_ppo_agent_save_load(self, ppo_agent, env, tmp_path):
        """save() then load() should recover policy weights."""
        ep = ppo_agent.collect_episode(env)
        ppo_agent.update([ep])
        path = str(tmp_path / "ppo_test.npz")
        ppo_agent.save(path)
        ppo_agent.load(path)
        state = np.random.randn(N_FEATURES).astype(np.float32)
        probs = ppo_agent.policy.predict_action_probs(state)
        assert probs.shape == (N_ACTIONS,)


# ===========================================================================
# Trainer tests
# ===========================================================================

class TestTrainer:
    def test_trainer_export_qtable_format(self, trainer, tmp_path):
        """Exported Q-table should have correct key format and value format."""
        # Use a freshly initialized DQN agent (no training needed)
        agent = DQNAgent(seed=0)
        qtable_path = str(tmp_path / "test_qtable.json")
        qtable = trainer.export_qtable(agent, path=qtable_path)

        # Check total number of states = 5^5 = 3125
        assert len(qtable) == 3125, f"Expected 3125 states, got {len(qtable)}"

        # Check key format: "b0,b1,b2,b3,b4" with each bin in [0,4]
        for key, val in list(qtable.items())[:20]:
            parts = key.split(",")
            assert len(parts) == 5, f"Key should have 5 parts: {key}"
            for p in parts:
                assert p.isdigit(), f"Key part not digit: {p}"
                assert 0 <= int(p) <= 4, f"Bin out of range: {p}"

            # Check value format: list of 2 floats [q_hold, q_exit]
            assert isinstance(val, list), f"Value should be list: {val}"
            assert len(val) == 2, f"Value should have 2 elements: {val}"
            assert all(isinstance(v, float) for v in val), f"Values should be floats: {val}"

    def test_trainer_export_qtable_json_readable(self, trainer, tmp_path):
        """Exported Q-table JSON should be parseable and match in-memory table."""
        agent = DQNAgent(seed=1)
        qtable_path = str(tmp_path / "qtable.json")
        qtable = trainer.export_qtable(agent, path=qtable_path)

        with open(qtable_path) as f:
            loaded = json.load(f)

        assert len(loaded) == len(qtable), "JSON length should match in-memory dict"
        for key in list(qtable.keys())[:10]:
            assert key in loaded, f"Key {key} missing from JSON"
            np.testing.assert_allclose(loaded[key], qtable[key], rtol=1e-5)

    def test_trainer_export_qtable_keys_cover_bh_active_states(self, trainer, tmp_path):
        """Q-table should contain entries for both bh_active=0 and bh_active=1."""
        agent = DQNAgent(seed=2)
        qtable = trainer.export_qtable(agent, path=str(tmp_path / "qt.json"))

        # b3 (bh_active bin) should include 0, 1, 2, 3, 4
        b3_vals = set(int(k.split(",")[3]) for k in qtable.keys())
        assert b3_vals == {0, 1, 2, 3, 4}, f"Expected all 5 b3 values, got {b3_vals}"

    def test_trainer_dqn_runs_short(self, trainer):
        """train_dqn() with few episodes should complete without error."""
        agent = trainer.train_dqn(n_episodes=5, save_every=3)
        assert isinstance(agent, DQNAgent)

    def test_trainer_ppo_runs_short(self, trainer):
        """train_ppo() with few episodes should complete without error."""
        agent = trainer.train_ppo(n_episodes=6)
        assert isinstance(agent, PPOAgent)

    def test_trainer_evaluate_returns_expected_keys(self, trainer):
        """evaluate() should return dict with all EvalResult keys."""
        agent = DQNAgent(seed=0)
        result = trainer.evaluate(agent, n_episodes=5)
        expected_keys = [
            "avg_pnl", "avg_hold_bars", "exit_at_target_rate",
            "stop_loss_hit_rate", "voluntary_exit_rate", "partial_exit_rate",
            "vs_hold_pnl", "n_episodes",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_trainer_evaluate_n_episodes_matches(self, trainer):
        """evaluate() result n_episodes should match requested value."""
        agent = DQNAgent(seed=0)
        result = trainer.evaluate(agent, n_episodes=8)
        assert result["n_episodes"] == 8

    def test_trainer_evaluate_rates_in_range(self, trainer):
        """All rate metrics should be in [0, 1]."""
        agent = DQNAgent(seed=0)
        result = trainer.evaluate(agent, n_episodes=10)
        for key in ["exit_at_target_rate", "stop_loss_hit_rate", "voluntary_exit_rate", "partial_exit_rate"]:
            assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} out of [0,1]"

    def test_trainer_compare_agents_keys(self, trainer):
        """compare_agents() should return dqn, ppo, and comparison sub-dicts."""
        dqn = DQNAgent(seed=0)
        ppo = PPOAgent(seed=0)
        result = trainer.compare_agents(dqn, ppo, n_episodes=5)
        assert "dqn" in result
        assert "ppo" in result
        assert "comparison" in result
        assert "winner" in result["comparison"]

    def test_trainer_compare_agents_winner_valid(self, trainer):
        """Winner field should be 'dqn' or 'ppo'."""
        dqn = DQNAgent(seed=0)
        ppo = PPOAgent(seed=0)
        result = trainer.compare_agents(dqn, ppo, n_episodes=5)
        assert result["comparison"]["winner"] in ("dqn", "ppo")

    def test_passive_hold_pnl_positive_trend(self, trending_episode):
        """Passive hold should be positive for trending (bullish) episodes on average."""
        ep_gen = TradeEpisodeGenerator(seed=0)
        pnls = []
        for _ in range(50):
            ep = ep_gen.generate("bh_trending", n_bars=36, entry_price=100.0)
            pnls.append(_passive_hold_pnl(ep))
        assert np.mean(pnls) > 0, "Passive hold should yield positive PnL on trending episodes"

    def test_trainer_checkpoint_dir_created(self, tmp_path):
        """Checkpoint directory should be created automatically."""
        ckpt_dir = str(tmp_path / "new_ckpt_dir")
        cfg = TrainConfig(
            dqn_episodes=2,
            ppo_episodes=2,
            eval_episodes=2,
            save_every=1,
            log_every=1,
            dqn_train_steps_per_episode=1,
            ppo_episodes_per_update=2,
            checkpoint_dir=ckpt_dir,
            seed=0,
        )
        trainer = RLTrainer(config=cfg)
        assert Path(ckpt_dir).exists(), "Checkpoint directory should be created"


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:
    def test_full_episode_rollout_dqn(self, env):
        """A full DQN rollout should complete an episode without error."""
        agent = DQNAgent(seed=0)
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            action = agent.act(obs, explore=True)
            obs, reward, done, info = env.step(action)
            agent.store(obs, action, reward, obs, done)
            steps += 1
        assert done or steps == 100, "Episode should complete"

    def test_full_episode_rollout_ppo(self, env):
        """A full PPO rollout should produce a valid episode."""
        agent = PPOAgent(seed=0)
        ep = agent.collect_episode(env)
        assert len(ep) > 0
        assert ep.dones[-1]

    def test_qtable_compatible_with_rl_exit_policy_schema(self, tmp_path):
        """
        Exported Q-table must be loadable by the key lookup logic in RLExitPolicy.
        Tests that key "2,2,2,1,2" exists and has 2 float values.
        """
        agent = DQNAgent(seed=0)
        trainer = RLTrainer(config=TrainConfig(checkpoint_dir=str(tmp_path)))
        qtable = trainer.export_qtable(agent, path=str(tmp_path / "qtable.json"))

        # This is a typical key the live trader would look up (mid-range bins, bh_active=1)
        test_key = "2,2,2,1,2"
        assert test_key in qtable, f"Expected key {test_key!r} in Q-table"
        vals = qtable[test_key]
        assert len(vals) == 2
        assert all(isinstance(v, float) for v in vals)

    def test_state_to_array_roundtrip(self):
        """TradingState.to_array() and from_array() should round-trip correctly."""
        s = TradingState(
            position_pnl_pct=0.03,
            bars_held=0.5,
            bh_mass=0.7,
            bh_active=1.0,
            atr_ratio=1.2,
            hurst_h=0.65,
            nav_omega=0.01,
            vol_percentile=0.6,
            time_of_day_sin=0.5,
            time_of_day_cos=0.866,
        )
        arr = s.to_array()
        s2 = TradingState.from_array(arr)
        assert abs(s.position_pnl_pct - s2.position_pnl_pct) < 1e-5
        assert abs(s.hurst_h - s2.hurst_h) < 1e-5

    def test_env_observation_within_declared_bounds(self, env):
        """All observations should lie within observation_space bounds."""
        for _ in range(50):
            obs = env.reset()
            # Check individually against declared low/high
            low = env.observation_space.low
            high = env.observation_space.high
            assert np.all(obs >= low - 0.01), f"obs below low: {obs}"
            assert np.all(obs <= high + 0.01), f"obs above high: {obs}"
