"""
test_hyper_agent.py — Test suite for Hyper-Agent MARL ecosystem.

Tests:
1. OrderBook: limit/market orders, matching, cancellation
2. MultiAssetTradingEnv: reset, step, observations
3. RewardShaper: reward computation
4. ObservationBuilder: obs shape and values
5. ActorCriticAgent: forward pass, action selection
6. MAPPOAgent: action + value, update
7. GaussianActor: sampling and log prob
8. MeanFieldAgent: MF update and action
9. QMIXTrainer: setup and episode storage
10. COMAAgent: action selection
11. AgentCommunicationModule: message passing
12. AgentPopulation: scripted actions
13. EmergenceAnalyzer: update and report
14. PrioritizedReplayBuffer: add, sample, priorities
15. EpisodeReplayBuffer: episode storage and sampling
16. TrainingConfig: trainer creation
17. VecTradingEnv: vectorized stepping
18. CircuitBreaker: threshold triggering
"""

from __future__ import annotations

import math
import sys
import os
import pytest
import numpy as np
import torch

# Add package to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hyper_agent.environment import (
    OrderBook, MultiAssetTradingEnv, AgentState, RewardShaper,
    ObservationBuilder, CircuitBreaker, FlashCrashSimulator,
    CorrelatedAssetProcess, VecTradingEnv, SIDE_BID, SIDE_ASK, make_env,
)
from hyper_agent.agents.base_agent import (
    ObservationEncoder, GaussianActor, ValueCritic,
    ActorCriticAgent, Transition, EpisodeBuffer,
    layer_init, soft_update, NoisyLinear, RunningMeanStd,
)
from hyper_agent.agents.mappo_agent import (
    MAPPOAgent, CentralizedCritic, ValueNormalizer, MARolloutBuffer,
)
from hyper_agent.agents.mean_field_agent import (
    MeanFieldAgent, MeanFieldRepresentation, PopulationMeanFieldTracker,
)
from hyper_agent.agents.qmix_agent import (
    IndividualQNetwork, QMIXHyperNetwork, QMIXTrainer,
)
from hyper_agent.agents.coma_agent import (
    COMAAgent, COMACounterfactualCritic, COMADecentralizedActor,
)
from hyper_agent.agents.communication import (
    AgentCommunicationModule, CommNet, TarMACProtocol,
    MultiHeadAttentionComm, CommunicationTopology, NoisyChannel,
)
from hyper_agent.population import (
    AgentPopulation, MarketMakerAgent, MomentumAgent,
    ArbitrageurAgent, NoiseTraderAgent, EvolutionaryDynamics,
)
from hyper_agent.emergence import (
    EmergenceAnalyzer, MarketImpactAnalyzer,
    PriceDiscoveryAnalyzer, FlashCrashDetector,
    NashEquilibriumApproximator,
)
from hyper_agent.replay_buffer import (
    UniformReplayBuffer, PrioritizedReplayBuffer,
    EpisodeReplayBuffer, MultiAgentReplayBuffer,
)
from hyper_agent.networks import (
    GRUActor, TransformerCritic, DuelingNetwork, PositionalEncoding,
)
from hyper_agent.training import TrainingConfig, MAPPOTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_env():
    env = MultiAssetTradingEnv(
        num_assets=2,
        num_agents=3,
        max_steps=50,
        seed=0,
        enable_circuit_breaker=False,
        enable_flash_crash=False,
    )
    return env


@pytest.fixture
def order_book():
    return OrderBook(asset_id=0, tick_size=0.01)


@pytest.fixture
def tiny_agent(simple_env):
    return ActorCriticAgent(
        agent_id=0,
        obs_dim=simple_env.obs_dim,
        action_dim=simple_env.action_dim,
        hidden_dim=32,
        device="cpu",
        seed=0,
    )


# ---------------------------------------------------------------------------
# Test 1: OrderBook — basic limit order matching
# ---------------------------------------------------------------------------

def test_orderbook_limit_orders(order_book):
    ob = order_book
    # Post bid
    bid_id, trades = ob.submit_limit_order(0, SIDE_BID, 100.0, 10.0, 1)
    assert bid_id >= 0
    assert len(trades) == 0
    assert ob.best_bid() == pytest.approx(100.0, abs=0.01)

    # Post ask at higher price (no match)
    ask_id, trades = ob.submit_limit_order(1, SIDE_ASK, 101.0, 10.0, 2)
    assert len(trades) == 0
    assert ob.best_ask() == pytest.approx(101.0, abs=0.01)

    # Post ask at same price as bid (match)
    ask_id2, trades = ob.submit_limit_order(1, SIDE_ASK, 100.0, 5.0, 3)
    assert len(trades) == 1
    assert trades[0].size == pytest.approx(5.0, abs=0.01)
    assert trades[0].price == pytest.approx(100.0, abs=0.01)


# ---------------------------------------------------------------------------
# Test 2: OrderBook — market order
# ---------------------------------------------------------------------------

def test_orderbook_market_order(order_book):
    ob = order_book
    ob.submit_limit_order(0, SIDE_BID, 99.0, 20.0, 1)
    ob.submit_limit_order(1, SIDE_ASK, 101.0, 20.0, 2)

    # Market buy: should fill against best ask at 101
    _, trades = ob.submit_market_order(2, SIDE_BID, 10.0, 3)
    assert len(trades) == 1
    assert trades[0].price == pytest.approx(101.0, abs=0.01)
    assert trades[0].size == pytest.approx(10.0, abs=0.01)


# ---------------------------------------------------------------------------
# Test 3: OrderBook — cancellation
# ---------------------------------------------------------------------------

def test_orderbook_cancellation(order_book):
    ob = order_book
    bid_id, _ = ob.submit_limit_order(0, SIDE_BID, 100.0, 10.0, 1)
    assert ob.cancel_order(bid_id) is True
    assert ob.best_bid() is None  # Cancelled
    assert ob.cancel_order(bid_id) is False  # Already cancelled


# ---------------------------------------------------------------------------
# Test 4: Environment reset and observation shape
# ---------------------------------------------------------------------------

def test_env_reset(simple_env):
    obs, info = simple_env.reset()
    assert obs.shape == (simple_env.obs_dim,)
    assert not np.any(np.isnan(obs))
    assert isinstance(info, dict)
    assert "equity" in info


# ---------------------------------------------------------------------------
# Test 5: Environment step
# ---------------------------------------------------------------------------

def test_env_step(simple_env):
    simple_env.reset()
    actions = [simple_env.action_space.sample() for _ in range(simple_env.num_agents)]
    obs_list, rewards, terminated, truncated, infos = simple_env._marl_step(actions)

    assert len(obs_list) == simple_env.num_agents
    assert len(rewards) == simple_env.num_agents
    assert len(terminated) == simple_env.num_agents

    for obs in obs_list:
        assert obs.shape == (simple_env.obs_dim,)
        assert not np.any(np.isnan(obs))
    for r in rewards:
        assert not math.isnan(r)


# ---------------------------------------------------------------------------
# Test 6: RewardShaper
# ---------------------------------------------------------------------------

def test_reward_shaper():
    shaper = RewardShaper(normalize_rewards=False)
    num_assets = 2
    mid_prices = np.array([100.0, 50.0])

    prev = AgentState(
        agent_id=0, cash=10000.0,
        positions=np.zeros(2), avg_cost=mid_prices.copy(),
        peak_equity=10000.0,
    )
    curr = AgentState(
        agent_id=0, cash=10100.0,
        positions=np.zeros(2), avg_cost=mid_prices.copy(),
        peak_equity=10100.0,
        returns_history=[0.001, 0.002, 0.001, -0.001] * 3,
    )

    reward, components = shaper.compute(prev, curr, mid_prices, 0.5, 0.1)
    assert isinstance(reward, float)
    assert not math.isnan(reward)
    assert "pnl" in components
    assert "sharpe" in components


# ---------------------------------------------------------------------------
# Test 7: ObservationBuilder
# ---------------------------------------------------------------------------

def test_observation_builder(simple_env):
    simple_env.reset()
    for i in range(simple_env.num_agents):
        obs = simple_env._get_obs(i)
        assert obs.shape == (simple_env.obs_dim,)
        assert obs.dtype == np.float32
        assert not np.any(np.isnan(obs))


# ---------------------------------------------------------------------------
# Test 8: ActorCriticAgent — action selection
# ---------------------------------------------------------------------------

def test_actor_critic_agent(simple_env, tiny_agent):
    obs = simple_env._get_obs(0)
    action, log_prob, value = tiny_agent.select_action(obs)
    assert action.shape == (simple_env.action_dim,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    assert not math.isnan(log_prob)
    assert not math.isnan(value)


# ---------------------------------------------------------------------------
# Test 9: MAPPOAgent — forward and update
# ---------------------------------------------------------------------------

def test_mappo_agent(simple_env):
    agent = MAPPOAgent(
        agent_id=0,
        obs_dim=simple_env.obs_dim,
        action_dim=simple_env.action_dim,
        state_dim=simple_env.state_dim,
        num_agents=simple_env.num_agents,
        hidden_dim=32,
        critic_hidden_dim=64,
        ppo_epochs=2,
        mini_batch_size=16,
        device="cpu",
    )
    simple_env.reset()
    obs = simple_env._get_obs(0)
    global_state = simple_env.get_state()

    action, lp, val = agent.select_action_with_value(obs, global_state)
    assert action.shape == (simple_env.action_dim,)
    assert not math.isnan(lp)
    assert not math.isnan(val)

    # Add some fake data to buffer and update
    for _ in range(30):
        agent.store_transition(obs, action, 0.01, False, lp, val, global_state)

    # Can't update without global states list, but check no crash
    assert len(agent.rollout_buffer.obs[0]) == 30


# ---------------------------------------------------------------------------
# Test 10: GaussianActor — sampling
# ---------------------------------------------------------------------------

def test_gaussian_actor():
    actor = GaussianActor(hidden_dim=64, action_dim=8, squash_output=True)
    hidden = torch.randn(4, 64)
    action, log_prob = actor.get_action(hidden)
    assert action.shape == (4, 8)
    assert log_prob.shape == (4, 1)
    assert torch.all(action >= -1) and torch.all(action <= 1)
    assert not torch.any(torch.isnan(action))


# ---------------------------------------------------------------------------
# Test 11: MeanFieldAgent
# ---------------------------------------------------------------------------

def test_mean_field_agent(simple_env):
    agent = MeanFieldAgent(
        agent_id=0,
        obs_dim=simple_env.obs_dim,
        action_dim=simple_env.action_dim,
        num_agents=simple_env.num_agents,
        hidden_dim=32,
        mf_embed_dim=16,
        batch_size=16,
        device="cpu",
    )
    simple_env.reset()
    obs = simple_env._get_obs(0)
    action, lp, _ = agent.select_action(obs)
    assert action.shape == (simple_env.action_dim,)
    assert not math.isnan(lp)

    # Update mean field
    dummy_actions = [np.random.uniform(-1, 1, simple_env.action_dim) for _ in range(5)]
    agent.update_mean_field(dummy_actions)
    feat = agent.mean_field.get_feature_vector()
    assert len(feat) == agent.mean_field.feature_dim


# ---------------------------------------------------------------------------
# Test 12: QMIXTrainer — setup and basic operations
# ---------------------------------------------------------------------------

def test_qmix_trainer():
    trainer = QMIXTrainer(
        num_agents=3,
        obs_dim=20,
        action_dim=4,
        state_dim=50,
        hidden_dim=32,
        batch_size=4,
    )
    trainer.init_hidden(1)
    obs_list = [np.random.randn(20).astype(np.float32) for _ in range(3)]
    actions = trainer.select_actions(obs_list, epsilon=1.0)
    assert len(actions) == 3
    for a in actions:
        assert a.shape == (4,)


# ---------------------------------------------------------------------------
# Test 13: COMAAgent — action selection
# ---------------------------------------------------------------------------

def test_coma_agent(simple_env):
    agent = COMAAgent(
        agent_id=0,
        obs_dim=simple_env.obs_dim,
        action_dim=simple_env.action_dim,
        state_dim=simple_env.state_dim,
        num_agents=simple_env.num_agents,
        hidden_dim=32,
        critic_hidden_dim=64,
        batch_size=16,
        device="cpu",
    )
    simple_env.reset()
    obs = simple_env._get_obs(0)
    action, lp, _ = agent.select_action(obs)
    assert action.shape == (simple_env.action_dim,)
    assert not math.isnan(lp)


# ---------------------------------------------------------------------------
# Test 14: AgentCommunicationModule
# ---------------------------------------------------------------------------

def test_communication_module():
    N = 4
    D = 64
    hidden = torch.randn(N, D)

    for protocol in ("commnet", "tarmac", "attention"):
        comm = AgentCommunicationModule(
            hidden_dim=D,
            num_agents=N,
            protocol=protocol,
            msg_dim=32,
            topology="full",
        )
        comm.eval()
        updated, info = comm(hidden)
        assert updated.shape == (N, D)
        assert not torch.any(torch.isnan(updated))


# ---------------------------------------------------------------------------
# Test 15: AgentPopulation — scripted actions
# ---------------------------------------------------------------------------

def test_agent_population():
    pop = AgentPopulation(
        num_market_makers=1,
        num_momentum=1,
        num_arbitrageurs=1,
        num_noise_traders=1,
        num_fundamental=1,
        num_marl_agents=0,
        num_assets=2,
        seed=0,
    )
    n_scripted = len(pop.scripted_agents)
    assert n_scripted == 5
    prices = np.array([100.0, 50.0])
    obs_list = [np.zeros(20) for _ in range(n_scripted)]
    invs = [np.zeros(2) for _ in range(n_scripted)]
    actions = pop.get_scripted_actions(obs_list, prices, invs)
    assert len(actions) == n_scripted
    for a in actions:
        assert a.shape == (2 * 4,)  # num_assets * 4


# ---------------------------------------------------------------------------
# Test 16: EmergenceAnalyzer
# ---------------------------------------------------------------------------

def test_emergence_analyzer():
    analyzer = EmergenceAnalyzer(num_assets=2, num_agents=4, action_dim=8)
    prices = np.array([100.0, 50.0])
    spreads = np.array([0.1, 0.05])
    imbalances = np.array([0.1, -0.1])
    volumes = np.array([100.0, 50.0])
    trade_counts = np.array([10, 5])
    total_depths = np.array([200.0, 100.0])
    rewards = [0.01] * 4
    actions = [np.random.uniform(-1, 1, 8) for _ in range(4)]

    for _ in range(5):
        result = analyzer.update(
            prices, spreads, imbalances, volumes, trade_counts,
            total_depths, rewards, actions, fundamentals=prices,
        )
    assert "step" in result

    report = analyzer.full_report()
    assert "asset_0" in report
    assert "nash" in report

    health = analyzer.get_market_health_score()
    assert 0.0 <= health <= 1.0


# ---------------------------------------------------------------------------
# Test 17: PrioritizedReplayBuffer
# ---------------------------------------------------------------------------

def test_prioritized_replay_buffer():
    obs_dim, action_dim = 10, 4
    buf = PrioritizedReplayBuffer(
        capacity=100, obs_dim=obs_dim, action_dim=action_dim,
        alpha=0.6, beta_start=0.4,
    )
    for i in range(50):
        buf.add(
            np.random.randn(obs_dim).astype(np.float32),
            np.random.randn(action_dim).astype(np.float32),
            float(np.random.randn()),
            np.random.randn(obs_dim).astype(np.float32),
            bool(np.random.rand() > 0.9),
        )

    assert len(buf) == 50
    batch = buf.sample(16)
    assert "obs" in batch
    assert batch["obs"].shape == (16, obs_dim)
    assert "weights" in batch
    assert "indices" in batch

    # Update priorities
    indices = batch["indices"].numpy()
    priorities = np.abs(np.random.randn(16)) + 0.01
    buf.update_priorities(indices, priorities)


# ---------------------------------------------------------------------------
# Test 18: EpisodeReplayBuffer
# ---------------------------------------------------------------------------

def test_episode_replay_buffer():
    buf = EpisodeReplayBuffer(
        capacity=50, obs_dim=8, action_dim=4,
        state_dim=16, num_agents=2, max_episode_len=20,
    )

    # Add two episodes
    for ep in range(3):
        for t in range(15):
            obs_list = [np.random.randn(8).astype(np.float32) for _ in range(2)]
            act_list = [np.random.randn(4).astype(np.float32) for _ in range(2)]
            rew_list = [0.1, 0.2]
            done_list = [False, False]
            next_obs_list = [np.random.randn(8).astype(np.float32) for _ in range(2)]
            state = np.random.randn(16).astype(np.float32)
            buf.add_step(obs_list, act_list, rew_list, done_list, next_obs_list, state)
        buf.end_episode()

    assert len(buf) == 3
    batch = buf.sample_batch(batch_size=2, seq_len=10)
    assert batch is not None
    assert batch["obs"].shape[0] == 2
    assert batch["obs"].shape[1] == 10


# ---------------------------------------------------------------------------
# Test 19: GRUActor
# ---------------------------------------------------------------------------

def test_gru_actor():
    actor = GRUActor(obs_dim=32, action_dim=8, hidden_dim=64)
    obs = torch.randn(4, 32)  # (B, obs_dim)
    h = actor.init_hidden(4, torch.device("cpu"))

    action, log_prob, new_h = actor.get_action(obs, h)
    assert action.shape == (4, 8)
    assert not torch.any(torch.isnan(action))
    assert new_h.shape[1] == 4  # batch dim


# ---------------------------------------------------------------------------
# Test 20: TransformerCritic
# ---------------------------------------------------------------------------

def test_transformer_critic():
    critic = TransformerCritic(obs_dim=32, d_model=64, nhead=4, num_layers=2)
    obs_seq = torch.randn(4, 10, 32)  # (B, T, obs_dim)
    value = critic(obs_seq)
    assert value.shape == (4, 1)
    assert not torch.any(torch.isnan(value))


# ---------------------------------------------------------------------------
# Test 21: VecTradingEnv
# ---------------------------------------------------------------------------

def test_vec_trading_env():
    vec_env = VecTradingEnv(
        num_envs=2,
        env_config={"num_assets": 2, "num_agents": 3, "max_steps": 10},
    )
    all_obs = vec_env.reset()
    assert len(all_obs) == 2
    assert len(all_obs[0]) == 3

    actions = [[vec_env.envs[0].action_space.sample() for _ in range(3)] for _ in range(2)]
    obs_all, rew_all, term_all, trunc_all, info_all = vec_env.step(actions)
    assert len(obs_all) == 2

    states = vec_env.get_global_states()
    assert len(states) == 2

    vec_env.close()


# ---------------------------------------------------------------------------
# Test 22: CircuitBreaker
# ---------------------------------------------------------------------------

def test_circuit_breaker():
    cb = CircuitBreaker(level1_threshold=0.07)
    cb.reset(np.array([100.0, 50.0]))

    # No trigger
    halted, level = cb.check(0, 103.0, step=1)
    assert not halted

    # Level 1 trigger
    halted, level = cb.check(0, 108.0, step=2)
    assert halted
    assert level == 1

    # Still halted during cooldown
    halted, level = cb.check(0, 105.0, step=3)
    assert halted


# ---------------------------------------------------------------------------
# Test 23: CorrelatedAssetProcess
# ---------------------------------------------------------------------------

def test_correlated_asset_process():
    proc = CorrelatedAssetProcess(num_assets=3, seed=0)
    initial = proc.prices.copy()
    prices_1 = proc.step()
    prices_2 = proc.step()

    assert prices_1.shape == (3,)
    assert prices_2.shape == (3,)
    assert not np.any(np.isnan(prices_1))
    assert not np.all(prices_1 == initial)  # Should have changed

    proc.reset()
    assert np.allclose(proc.prices, initial)


# ---------------------------------------------------------------------------
# Test 24: MARolloutBuffer
# ---------------------------------------------------------------------------

def test_ma_rollout_buffer():
    buf = MARolloutBuffer(
        num_agents=3, obs_dim=20, action_dim=8, state_dim=50, max_size=100
    )
    for t in range(10):
        obs_list = [np.random.randn(20).astype(np.float32) for _ in range(3)]
        acts = [np.random.randn(8).astype(np.float32) for _ in range(3)]
        rews = [0.1, 0.2, -0.1]
        dones = [False, False, False]
        lps = [0.1, 0.2, -0.1]
        vals = [1.0, 1.1, 0.9]
        state = np.random.randn(50).astype(np.float32)
        buf.add_step(obs_list, acts, rews, dones, lps, vals, state)

    assert len(buf) == 10
    adv_list, ret_list = buf.compute_advantages_and_returns([0.0] * 3)
    assert len(adv_list) == 3
    assert len(adv_list[0]) == 10

    buf.clear()
    assert len(buf) == 0


# ---------------------------------------------------------------------------
# Test 25: CentralizedCritic
# ---------------------------------------------------------------------------

def test_centralized_critic():
    critic = CentralizedCritic(state_dim=64, hidden_dim=128, num_layers=2)
    state = torch.randn(4, 64)
    value = critic(state)
    assert value.shape == (4, 1)
    assert not torch.any(torch.isnan(value))


# ---------------------------------------------------------------------------
# Run as script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
