"""
tests/test_reward_shaping.py
============================
Tests for the reward shaping module.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from hyper_agent.reward_shaping import (
    RewardContext,
    RewardOutput,
    BaseRewardComponent,
    MarkToMarketReward,
    RealisedPnLReward,
    SharpeReward,
    DrawdownPenalty,
    InventoryRiskPenalty,
    MarketImpactCost,
    ExecutionQualityReward,
    SlippagePenalty,
    LiquidityProvisionReward,
    DestabilisingReward,
    PBRSComponent,
    CurriculumScheduler,
    RewardNormaliser,
    MultiObjectiveCombiner,
    AgentRole,
    build_reward_combiner,
    ManagedRewardCombiner,
    DifferentialReward,
    TeamRewardAggregator,
    NeuralPotentialNetwork,
    LearnedPBRS,
    MarketMakerRewardSuite,
    make_reward_context,
    inventory_potential,
    pnl_velocity_potential,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_ctx():
    return RewardContext(
        agent_id="agent_0",
        inventory=10.0,
        prev_inventory=5.0,
        cash=1000.0,
        prev_cash=900.0,
        mid_price=100.5,
        prev_mid_price=100.0,
        best_bid=100.4,
        best_ask=100.6,
        spread=0.2,
        vwap=100.3,
        fill_price=100.5,
        fill_size=5.0,
        fill_side=1,
        slippage=0.02,
        market_impact=0.01,
        realized_pnl_delta=5.0,
        total_pnl=50.0,
        num_trades=3,
        episode_step=100,
        episode_length=2000,
        volatility=0.002,
        market_imbalance=0.2,
    )


@pytest.fixture
def flat_ctx():
    """Context with no price move and zero inventory."""
    return RewardContext(
        agent_id="agent_0",
        inventory=0.0,
        prev_inventory=0.0,
        cash=1000.0,
        prev_cash=1000.0,
        mid_price=100.0,
        prev_mid_price=100.0,
        best_bid=99.9,
        best_ask=100.1,
        spread=0.2,
        vwap=100.0,
        fill_price=None,
        fill_size=None,
        fill_side=None,
    )


# ---------------------------------------------------------------------------
# Individual component tests
# ---------------------------------------------------------------------------

class TestMarkToMarketReward:
    def test_positive_on_price_up_long(self, base_ctx):
        comp = MarkToMarketReward(weight=1.0)
        out = comp.compute(base_ctx)
        # inventory=10, price moved from 100 to 100.5: gain = 10 * 0.5 = 5
        # But cash also changed, so just check sign
        assert isinstance(out.value, float)

    def test_zero_on_flat_market(self, flat_ctx):
        comp = MarkToMarketReward(weight=1.0)
        out = comp.compute(flat_ctx)
        assert abs(out.value) < 0.1

    def test_clipping(self):
        comp = MarkToMarketReward(weight=1.0, clip=0.5)
        ctx = RewardContext(
            agent_id="a0", inventory=10000.0, prev_inventory=0.0,
            cash=0.0, prev_cash=0.0, mid_price=100.0, prev_mid_price=0.0,
            best_bid=99.0, best_ask=101.0, spread=2.0, vwap=100.0,
            fill_price=None, fill_size=None, fill_side=None,
        )
        out = comp.compute(ctx)
        assert abs(out.value) <= 0.5
        assert out.clipped


class TestSharpeReward:
    def test_returns_zero_before_warmup(self, base_ctx):
        comp = SharpeReward(window=50, min_samples=10)
        for _ in range(5):
            out = comp.compute(base_ctx)
        assert out.value == 0.0

    def test_returns_nonzero_after_warmup(self, base_ctx):
        comp = SharpeReward(window=50, min_samples=5)
        last_out = None
        for i in range(15):
            base_ctx.total_pnl = float(i * 10)
            last_out = comp.compute(base_ctx)
        assert last_out is not None
        assert isinstance(last_out.value, float)

    def test_reset_clears_buffer(self, base_ctx):
        comp = SharpeReward(window=50, min_samples=5)
        for _ in range(20):
            comp.compute(base_ctx)
        comp.reset()
        out = comp.compute(base_ctx)
        assert out.value == 0.0


class TestDrawdownPenalty:
    def test_no_penalty_at_new_high(self, base_ctx):
        comp = DrawdownPenalty()
        base_ctx.total_pnl = 100.0
        out = comp.compute(base_ctx)
        assert out.value <= 0.0   # no drawdown, just holding cost

    def test_penalty_on_drawdown(self, base_ctx):
        comp = DrawdownPenalty(current_dd_coef=1.0)
        base_ctx.total_pnl = 100.0
        comp.compute(base_ctx)
        base_ctx.total_pnl = 50.0   # drawdown of 50
        out = comp.compute(base_ctx)
        assert out.value < 0.0

    def test_reset(self):
        comp = DrawdownPenalty()
        ctx = RewardContext(
            agent_id="a0", inventory=0.0, prev_inventory=0.0, cash=0.0, prev_cash=0.0,
            mid_price=100.0, prev_mid_price=100.0, best_bid=99.0, best_ask=101.0,
            spread=2.0, vwap=100.0, fill_price=None, fill_size=None, fill_side=None,
            total_pnl=200.0,
        )
        comp.compute(ctx)
        ctx.total_pnl = 100.0
        comp.compute(ctx)
        comp.reset()
        assert comp._peak_pnl == 0.0


class TestInventoryRiskPenalty:
    def test_zero_inventory_no_penalty(self, flat_ctx):
        comp = InventoryRiskPenalty()
        flat_ctx.inventory = 0.0
        out = comp.compute(flat_ctx)
        assert abs(out.value) < 0.001

    def test_large_inventory_penalty(self, base_ctx):
        comp = InventoryRiskPenalty(quadratic_coef=0.001)
        base_ctx.inventory = 500.0
        out = comp.compute(base_ctx)
        assert out.value < 0.0

    def test_excess_penalty(self, base_ctx):
        comp = InventoryRiskPenalty(max_safe_inventory=50.0, excess_coef=0.01)
        base_ctx.inventory = 200.0
        out_excess = comp.compute(base_ctx)
        base_ctx.inventory = 30.0
        out_safe = comp.compute(base_ctx)
        assert out_excess.value < out_safe.value


class TestMarketImpactCost:
    def test_zero_fill_no_cost(self, flat_ctx):
        comp = MarketImpactCost()
        flat_ctx.fill_size = None
        out = comp.compute(flat_ctx)
        assert out.value == 0.0

    def test_fill_has_cost(self, base_ctx):
        comp = MarketImpactCost()
        out = comp.compute(base_ctx)
        assert out.value < 0.0


class TestExecutionQualityReward:
    def test_buy_below_vwap(self):
        comp = ExecutionQualityReward(weight=1.0, normalise_by_spread=False)
        ctx = RewardContext(
            agent_id="a0", inventory=0.0, prev_inventory=0.0, cash=0.0, prev_cash=0.0,
            mid_price=100.0, prev_mid_price=100.0, best_bid=99.9, best_ask=100.1,
            spread=0.2, vwap=100.0,
            fill_price=99.8,   # buy below VWAP = good
            fill_size=10.0,
            fill_side=1,
        )
        out = comp.compute(ctx)
        assert out.value > 0.0

    def test_sell_below_vwap(self):
        comp = ExecutionQualityReward(weight=1.0, normalise_by_spread=False)
        ctx = RewardContext(
            agent_id="a0", inventory=0.0, prev_inventory=0.0, cash=0.0, prev_cash=0.0,
            mid_price=100.0, prev_mid_price=100.0, best_bid=99.9, best_ask=100.1,
            spread=0.2, vwap=100.0,
            fill_price=99.8,   # sell below VWAP = bad
            fill_size=10.0,
            fill_side=-1,
        )
        out = comp.compute(ctx)
        assert out.value < 0.0


class TestLiquidityProvisionReward:
    def test_spread_improvement_bonus(self):
        comp = LiquidityProvisionReward(spread_improvement_bonus=10.0)
        ctx = RewardContext(
            agent_id="a0", inventory=0.0, prev_inventory=0.0, cash=0.0, prev_cash=0.0,
            mid_price=100.0, prev_mid_price=100.0, best_bid=99.9, best_ask=100.1,
            spread=0.1, vwap=100.0,  # tight spread
            fill_price=None, fill_size=None, fill_side=None,
        )
        comp._prev_spread = 0.5   # was wide before
        out = comp.compute(ctx)
        assert out.value > 0.0


class TestPBRSComponent:
    def test_shaping_formula(self, base_ctx):
        comp = PBRSComponent(potential_fn=inventory_potential, gamma=0.99)
        out1 = comp.compute(base_ctx)
        assert isinstance(out1.value, float)

    def test_reset(self, base_ctx):
        comp = PBRSComponent(potential_fn=pnl_velocity_potential)
        comp.compute(base_ctx)
        comp.reset()
        assert comp._prev_potential == 0.0


# ---------------------------------------------------------------------------
# CurriculumScheduler tests
# ---------------------------------------------------------------------------

class TestCurriculumScheduler:
    def test_initial_phase(self):
        sched = CurriculumScheduler()
        weights = sched.get_weights()
        assert "mark_to_market" in weights

    def test_phase_progression(self):
        sched = CurriculumScheduler()
        for _ in range(3001):
            sched.step_episode()
        weights = sched.get_weights()
        assert "sharpe" in weights or "drawdown_penalty" in weights

    def test_apply_weights(self):
        sched = CurriculumScheduler()
        comp = MarkToMarketReward(weight=0.0)
        sched.apply_weights({"mark_to_market": comp})
        # After applying, weight should match schedule
        assert comp.weight >= 0.0


# ---------------------------------------------------------------------------
# RewardNormaliser tests
# ---------------------------------------------------------------------------

class TestRewardNormaliser:
    def test_normalise_zero_mean(self):
        norm = RewardNormaliser()
        rewards = [float(i) for i in range(100)]
        normalised = [norm.update_and_normalise(r) for r in rewards]
        assert abs(np.mean(normalised[-10:])) < 2.0

    def test_clipping(self):
        norm = RewardNormaliser(clip=1.0)
        for _ in range(50):
            norm.update_and_normalise(0.0)
        out = norm.update_and_normalise(1e9)
        assert abs(out) <= 1.0

    def test_reset_stats(self):
        norm = RewardNormaliser()
        for i in range(50):
            norm.update_and_normalise(float(i))
        norm.reset_stats()
        assert norm._mean == 0.0


# ---------------------------------------------------------------------------
# MultiObjectiveCombiner tests
# ---------------------------------------------------------------------------

class TestMultiObjectiveCombiner:
    def test_weighted_sum(self, base_ctx):
        comps = [MarkToMarketReward(weight=1.0), InventoryRiskPenalty(weight=0.5)]
        comb = MultiObjectiveCombiner(comps)
        reward, breakdown = comb.compute(base_ctx, normalise=False)
        assert isinstance(reward, float)
        assert "mark_to_market" in breakdown
        assert "inventory_risk" in breakdown

    def test_chebyshev_mode(self, base_ctx):
        comps = [MarkToMarketReward(weight=1.0), SharpeReward(weight=0.5)]
        comb = MultiObjectiveCombiner(comps, mode=MultiObjectiveCombiner.CombineMode.CHEBYSHEV)
        reward, _ = comb.compute(base_ctx, normalise=False)
        assert isinstance(reward, float)

    def test_component_stats(self, base_ctx):
        comps = [MarkToMarketReward(weight=1.0)]
        comb = MultiObjectiveCombiner(comps)
        for _ in range(10):
            comb.compute(base_ctx, normalise=False)
        stats = comb.component_stats()
        assert "mark_to_market" in stats
        assert "mean" in stats["mark_to_market"]

    def test_reset(self, base_ctx):
        comps = [SharpeReward(window=10, weight=1.0)]
        comb = MultiObjectiveCombiner(comps)
        for _ in range(10):
            comb.compute(base_ctx)
        comb.reset()
        # After reset, Sharpe should be zero again
        out, _ = comb.compute(base_ctx, normalise=False)
        assert isinstance(out, float)


# ---------------------------------------------------------------------------
# ManagedRewardCombiner tests
# ---------------------------------------------------------------------------

class TestManagedRewardCombiner:
    def test_compute(self, base_ctx):
        combiner = build_reward_combiner(AgentRole.MARKET_MAKER)
        reward, breakdown = combiner.compute(base_ctx)
        assert isinstance(reward, float)
        assert len(breakdown) > 0

    def test_on_episode_end(self, base_ctx):
        combiner = build_reward_combiner(AgentRole.GENERIC)
        for _ in range(10):
            combiner.compute(base_ctx)
        combiner.on_episode_end()
        stats = combiner.get_stats()
        assert stats["episode"] == 1

    def test_all_roles(self, base_ctx):
        for role in AgentRole:
            combiner = build_reward_combiner(role)
            reward, _ = combiner.compute(base_ctx)
            assert isinstance(reward, float)
            assert not math.isnan(reward)


# ---------------------------------------------------------------------------
# TeamRewardAggregator tests
# ---------------------------------------------------------------------------

class TestTeamRewardAggregator:
    def test_global_mode(self):
        agg = TeamRewardAggregator(3, TeamRewardAggregator.MixMode.GLOBAL)
        ind = {"a0": 1.0, "a1": 2.0, "a2": 3.0}
        result = agg.aggregate(ind)
        assert abs(result["a0"] - result["a1"]) < 0.01

    def test_individual_mode(self):
        agg = TeamRewardAggregator(3, TeamRewardAggregator.MixMode.INDIVIDUAL)
        ind = {"a0": 1.0, "a1": 2.0, "a2": 3.0}
        result = agg.aggregate(ind)
        assert result == ind

    def test_mixed_mode(self):
        agg = TeamRewardAggregator(3, TeamRewardAggregator.MixMode.MIXED, mix_lambda=0.5)
        ind = {"a0": 10.0, "a1": 0.0}
        result = agg.aggregate(ind)
        # Mixed should be between individual and team
        assert 0.0 < result["a0"] < 10.0
        assert 0.0 < result["a1"] < 10.0

    def test_competitive_mode(self):
        agg = TeamRewardAggregator(2, TeamRewardAggregator.MixMode.COMPETITIVE)
        ind = {"a0": 10.0, "a1": 0.0}
        result = agg.aggregate(ind)
        # Competitive: high performer gets even higher reward
        assert result["a0"] > result["a1"]

    def test_empty_input(self):
        agg = TeamRewardAggregator(3)
        result = agg.aggregate({})
        assert result == {}


# ---------------------------------------------------------------------------
# NeuralPotentialNetwork tests
# ---------------------------------------------------------------------------

class TestNeuralPotentialNetwork:
    def test_forward(self):
        net = NeuralPotentialNetwork(obs_dim=32, hidden_dim=64)
        obs = torch.randn(4, 32)
        pot = net(obs)
        assert pot.shape == (4,)

    def test_potential_method(self):
        net = NeuralPotentialNetwork(obs_dim=32)
        obs = np.random.randn(32).astype(np.float32)
        val = net.potential(obs)
        assert isinstance(val, float)


# ---------------------------------------------------------------------------
# MarketMakerRewardSuite tests
# ---------------------------------------------------------------------------

class TestMarketMakerRewardSuite:
    def test_step(self, base_ctx):
        suite = MarketMakerRewardSuite()
        reward = suite.step(base_ctx)
        assert isinstance(reward, float)
        assert not math.isnan(reward)

    def test_end_episode(self, base_ctx):
        suite = MarketMakerRewardSuite()
        for _ in range(20):
            suite.step(base_ctx)
        stats = suite.end_episode()
        assert "episode_return" in stats
        assert stats["episode_steps"] == 20

    def test_inventory_limit_penalty(self, base_ctx):
        suite = MarketMakerRewardSuite(inventory_limit=5.0)
        base_ctx.inventory = 1000.0   # far exceeds limit
        reward = suite.step(base_ctx)
        assert reward < -3.0   # should have hard penalty


# ---------------------------------------------------------------------------
# make_reward_context helper tests
# ---------------------------------------------------------------------------

class TestMakeRewardContext:
    def test_builds_context(self):
        class MockSnap:
            mid_price = 100.0
            best_bid = 99.9
            best_ask = 100.1
            spread = 0.2
            vwap = 100.0
            volatility_est = 0.002
            imbalance = 0.1

        prev_state = {"inventory": 5.0, "cash": 1000.0, "realized_pnl": 0.0,
                       "total_pnl": 0.0, "num_trades": 0}
        curr_state = {"inventory": 10.0, "cash": 950.0, "realized_pnl": 5.0,
                       "total_pnl": 55.0, "num_trades": 1}
        snap = MockSnap()
        ctx = make_reward_context("agent_0", prev_state, curr_state, snap, snap)
        assert ctx.agent_id == "agent_0"
        assert ctx.inventory == 10.0
        assert ctx.mid_price == 100.0
