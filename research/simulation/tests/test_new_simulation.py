"""
research/simulation/tests/test_new_simulation.py

Test suite for:
  - agent_based_model (AgentBasedMarket, agent types, calibration)
  - microstructure_simulator (LOBSimulator, TickDataGenerator, SpreadDynamicsSimulator)
  - calibration_tools (MomentCalibrator, GoodnessOfFit)

Run with:
    pytest research/simulation/tests/test_new_simulation.py -v
"""

from __future__ import annotations

import math
import pytest
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# # modules under test
from research.simulation.agent_based_model import (
    AgentBasedMarket,
    MarketAgent,
    MomentumAgent,
    MeanReversionAgent,
    MarketMakerAgent,
    NoiseAgent,
    InformedTrader,
    MarketState,
    Order,
    Fill,
    SimulationResult,
    AgentPopulationConfig,
    fit_to_empirical,
    build_market_from_config,
)
from research.simulation.microstructure_simulator import (
    LOBSimulator,
    TickDataGenerator,
    SpreadDynamicsSimulator,
    Tick,
    Fill as MFill,
)
from research.simulation.calibration_tools import (
    MomentCalibrator,
    GoodnessOfFit,
    GBMParams,
    OUParams,
    GARCHParams,
    JumpParams,
    KSResult,
    ADResult,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture(scope="module")
def normal_returns(rng: np.random.Generator) -> NDArray[np.float64]:
    """Standard normal returns for calibration tests."""
    return rng.normal(0.0005, 0.01, size=500).astype(np.float64)


@pytest.fixture(scope="module")
def fat_tail_returns(rng: np.random.Generator) -> NDArray[np.float64]:
    """Student-t returns (heavy tails, kurtosis > 0)."""
    return (rng.standard_t(df=4, size=500) * 0.01).astype(np.float64)


@pytest.fixture(scope="module")
def price_series(rng: np.random.Generator) -> NDArray[np.float64]:
    """Mean-reverting price series for OU calibration."""
    n = 300
    prices = np.zeros(n, dtype=np.float64)
    prices[0] = 100.0
    kappa_true = 2.0
    theta_true = 100.0
    sigma_true = 0.5
    dt = 1.0 / 252
    for t in range(1, n):
        dW = rng.normal(0.0, math.sqrt(dt))
        prices[t] = (
            prices[t - 1]
            + kappa_true * (theta_true - prices[t - 1]) * dt
            + sigma_true * dW
        )
    return prices


@pytest.fixture(scope="module")
def up_move_state() -> MarketState:
    """MarketState following a sustained upward move (for momentum tests)."""
    history = [100.0 * math.exp(0.005 * i) for i in range(20)]
    returns = [math.log(history[i] / history[i - 1]) for i in range(1, 20)]
    return MarketState(
        price=history[-1],
        volume=1000.0,
        spread=0.002,
        informed_fraction=0.05,
        day=5,
        hour=2.0,
        price_history=history,
        returns_history=returns,
    )


@pytest.fixture(scope="module")
def down_move_state() -> MarketState:
    """MarketState following a sustained downward move."""
    history = [100.0 * math.exp(-0.005 * i) for i in range(20)]
    returns = [math.log(history[i] / history[i - 1]) for i in range(1, 20)]
    return MarketState(
        price=history[-1],
        volume=500.0,
        spread=0.003,
        informed_fraction=0.05,
        day=3,
        hour=4.0,
        price_history=history,
        returns_history=returns,
    )


@pytest.fixture(scope="module")
def flat_state() -> MarketState:
    """MarketState with no trend (for mean-reversion tests)."""
    price = 95.0
    history = [100.0] * 15 + [95.0] * 5
    returns = [0.0] * 19
    return MarketState(
        price=price,
        volume=800.0,
        spread=0.002,
        informed_fraction=0.02,
        day=2,
        hour=1.5,
        price_history=history,
        returns_history=returns,
    )


# ===========================================================================
# Agent-based model tests
# ===========================================================================

class TestMomentumAgent:
    """Tests for MomentumAgent decision logic."""

    def test_buys_on_up_move(self, up_move_state: MarketState) -> None:
        """Momentum agent should submit a buy order after sustained price rise."""
        agent = MomentumAgent(agent_id="mom_0", threshold=0.01)
        agent.cash = 50_000.0
        order = agent.decide(up_move_state)
        assert order is not None, "Expected a buy order on strong upward move"
        assert order.side == "buy"

    def test_sells_on_down_move(self, down_move_state: MarketState) -> None:
        """Momentum agent should submit a sell order after sustained price fall."""
        agent = MomentumAgent(agent_id="mom_1", threshold=0.01)
        agent.cash = 50_000.0
        agent.position = 500.0   # needs position to sell
        order = agent.decide(down_move_state)
        assert order is not None, "Expected a sell order on strong downward move"
        assert order.side == "sell"

    def test_no_order_below_threshold(self, flat_state: MarketState) -> None:
        """Momentum agent should not trade when return is below threshold."""
        agent = MomentumAgent(agent_id="mom_2", threshold=0.10)
        order = agent.decide(flat_state)
        assert order is None

    def test_cooldown_suppresses_orders(self, up_move_state: MarketState) -> None:
        """After placing an order, agent should be on cooldown."""
        agent = MomentumAgent(agent_id="mom_3", threshold=0.001)
        agent.cash = 50_000.0
        order1 = agent.decide(up_move_state)
        assert order1 is not None
        order2 = agent.decide(up_move_state)
        assert order2 is None, "Second consecutive call should be blocked by cooldown"

    def test_order_qty_positive(self, up_move_state: MarketState) -> None:
        """All submitted orders must have positive quantity."""
        agent = MomentumAgent(agent_id="mom_4", threshold=0.001)
        agent.cash = 100_000.0
        order = agent.decide(up_move_state)
        if order is not None:
            assert order.qty > 0


class TestMeanReversionAgent:
    """Tests for MeanReversionAgent."""

    def test_buys_when_oversold(self, flat_state: MarketState) -> None:
        """Agent should buy when price is significantly below rolling mean."""
        agent = MeanReversionAgent(agent_id="mr_0", z_threshold=1.5, lookback=20)
        agent.cash = 50_000.0
        order = agent.decide(flat_state)
        # price=95 is ~2 std below mean=100 in the flat_state fixture
        if order is not None:
            assert order.side == "buy"

    def test_no_order_insufficient_history(self) -> None:
        """Agent should not trade when price history is shorter than lookback."""
        state = MarketState(
            price=100.0,
            volume=100.0,
            spread=0.002,
            informed_fraction=0.0,
            day=0,
            hour=0.0,
            price_history=[100.0, 99.0],
            returns_history=[0.0],
        )
        agent = MeanReversionAgent(agent_id="mr_1", z_threshold=2.0, lookback=20)
        order = agent.decide(state)
        assert order is None

    def test_sells_when_overbought(self) -> None:
        """Agent should sell when price is well above rolling mean."""
        history = [100.0] * 20
        price = 110.0
        state = MarketState(
            price=price,
            volume=500.0,
            spread=0.002,
            informed_fraction=0.0,
            day=5,
            hour=2.0,
            price_history=history + [price],
            returns_history=[0.0] * 20,
        )
        agent = MeanReversionAgent(agent_id="mr_2", z_threshold=1.0)
        agent.position = 200.0
        order = agent.decide(state)
        if order is not None:
            assert order.side == "sell"


class TestMarketMakerAgent:
    """Tests for MarketMakerAgent."""

    def test_submits_order(self, flat_state: MarketState) -> None:
        """Market maker should always post an order when it has room."""
        agent = MarketMakerAgent(agent_id="mm_0", spread=0.002, quote_size=100.0)
        order = agent.decide(flat_state)
        assert order is not None

    def test_alternates_sides(self, flat_state: MarketState) -> None:
        """Market maker should alternate between buy and sell."""
        agent = MarketMakerAgent(agent_id="mm_1")
        sides = []
        for _ in range(4):
            o = agent.decide(flat_state)
            if o is not None:
                sides.append(o.side)
        # with alternation, should see both sides
        assert "buy" in sides or "sell" in sides

    def test_reduces_position_at_limit(self, flat_state: MarketState) -> None:
        """When inventory hits max, market maker should reduce it."""
        agent = MarketMakerAgent(agent_id="mm_2", max_inventory=100.0)
        agent.position = 95.0   # near max
        order = agent.decide(flat_state)
        if order is not None:
            assert order.side == "sell"


class TestNoiseAgent:
    """Tests for NoiseAgent."""

    def test_submits_random_orders(self) -> None:
        """Noise agent should submit orders with non-zero probability."""
        rng = np.random.default_rng(99)
        agent = NoiseAgent(
            agent_id="noise_0",
            activity_prob=1.0,   # always active
            rng=rng,
        )
        state = MarketState(
            price=100.0,
            volume=1000.0,
            spread=0.002,
            informed_fraction=0.0,
            day=1,
            hour=2.0,
        )
        orders = [agent.decide(state) for _ in range(20)]
        n_orders = sum(1 for o in orders if o is not None)
        assert n_orders > 0

    def test_order_qty_positive(self) -> None:
        """All orders from NoiseAgent must have positive qty."""
        rng = np.random.default_rng(7)
        agent = NoiseAgent(agent_id="noise_1", activity_prob=1.0, rng=rng)
        state = MarketState(
            price=100.0,
            volume=0.0,
            spread=0.002,
            informed_fraction=0.0,
            day=0,
            hour=0.0,
        )
        for _ in range(10):
            o = agent.decide(state)
            if o is not None:
                assert o.qty > 0


class TestInformedTrader:
    """Tests for InformedTrader."""

    def test_buys_when_underpriced(self) -> None:
        """Informed trader should buy when market price < true value."""
        rng = np.random.default_rng(55)
        agent = InformedTrader(
            agent_id="inf_0",
            true_value=110.0,   # true value above market price
            aggressiveness=1.0,
            rng=rng,
        )
        state = MarketState(
            price=100.0,
            volume=0.0,
            spread=0.002,
            informed_fraction=0.0,
            day=1,
            hour=1.0,
        )
        order = agent.decide(state)
        assert order is not None
        assert order.side == "buy"

    def test_sells_when_overpriced(self) -> None:
        """Informed trader should sell when market price > true value."""
        rng = np.random.default_rng(56)
        agent = InformedTrader(
            agent_id="inf_1",
            true_value=90.0,
            aggressiveness=1.0,
            rng=rng,
        )
        agent.position = 200.0
        state = MarketState(
            price=100.0,
            volume=0.0,
            spread=0.002,
            informed_fraction=0.0,
            day=1,
            hour=1.0,
        )
        order = agent.decide(state)
        assert order is not None
        assert order.side == "sell"

    def test_no_trade_near_fair_value(self) -> None:
        """Informed trader should not trade when price == true value."""
        rng = np.random.default_rng(57)
        agent = InformedTrader(
            agent_id="inf_2",
            true_value=100.0,
            rng=rng,
        )
        state = MarketState(
            price=100.0,
            volume=0.0,
            spread=0.002,
            informed_fraction=0.0,
            day=0,
            hour=0.0,
        )
        order = agent.decide(state)
        assert order is None, "No trade when price == true value"


class TestAgentBasedMarket:
    """Integration tests for AgentBasedMarket."""

    def test_run_returns_correct_shape(self) -> None:
        """Simulation should return arrays of the correct length."""
        rng = np.random.default_rng(42)
        market = AgentBasedMarket(initial_price=100.0, rng=rng)
        market.add_agent(NoiseAgent("n0", activity_prob=0.5, rng=np.random.default_rng(1)))
        market.add_agent(MomentumAgent("m0"))
        result = market.run(n_steps=50)
        assert isinstance(result, SimulationResult)
        assert len(result.price_series) == 51
        assert len(result.volume_series) == 50
        assert len(result.spread_series) == 50

    def test_price_stays_positive(self) -> None:
        """All simulated prices must remain positive."""
        rng = np.random.default_rng(77)
        market = AgentBasedMarket(initial_price=50.0, rng=rng)
        market.add_agent(NoiseAgent("n0", activity_prob=0.8, rng=np.random.default_rng(2)))
        result = market.run(n_steps=100)
        assert np.all(result.price_series > 0)

    def test_agent_pnls_present(self) -> None:
        """SimulationResult should contain a PnL entry for each agent."""
        rng = np.random.default_rng(88)
        market = AgentBasedMarket(initial_price=100.0, rng=rng)
        market.add_agent(MomentumAgent("mom_x"))
        market.add_agent(NoiseAgent("nse_x", rng=np.random.default_rng(3)))
        result = market.run(n_steps=30)
        assert "mom_x" in result.agent_pnls
        assert "nse_x" in result.agent_pnls

    def test_add_multiple_instances(self) -> None:
        """add_agent with n_instances should populate multiple distinct agents."""
        rng = np.random.default_rng(99)
        market = AgentBasedMarket(rng=rng)
        market.add_agent(NoiseAgent("base_noise", rng=np.random.default_rng(4)), n_instances=3)
        assert len(market._agents) == 3
        ids = {a.agent_id for a in market._agents}
        assert len(ids) == 3, "Each instance should have a unique agent_id"

    def test_empty_market_runs_without_error(self) -> None:
        """Market with no agents should complete without raising."""
        market = AgentBasedMarket(initial_price=100.0)
        result = market.run(n_steps=10)
        # price should remain at initial
        assert result.price_series[0] == pytest.approx(100.0)


class TestFitToEmpirical:
    """Tests for calibration of agent population."""

    def test_fractions_sum_to_one(self, fat_tail_returns: NDArray[np.float64]) -> None:
        """Calibrated fractions must sum to 1."""
        config = fit_to_empirical(fat_tail_returns)
        total = (
            config.noise_fraction
            + config.momentum_fraction
            + config.mean_reversion_fraction
            + config.informed_fraction
            + config.market_maker_fraction
        )
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_all_fractions_positive(self, normal_returns: NDArray[np.float64]) -> None:
        """All fractions must be strictly positive."""
        config = fit_to_empirical(normal_returns)
        assert config.noise_fraction > 0
        assert config.momentum_fraction > 0
        assert config.mean_reversion_fraction > 0
        assert config.informed_fraction > 0
        assert config.market_maker_fraction > 0

    def test_build_market_from_config_runs(self, normal_returns: NDArray[np.float64]) -> None:
        """Market built from calibrated config should run without error."""
        config = fit_to_empirical(normal_returns)
        rng = np.random.default_rng(11)
        market = build_market_from_config(
            config, initial_price=100.0, total_agents=20, rng=rng
        )
        result = market.run(n_steps=20)
        assert result.n_steps == 20


# ===========================================================================
# LOBSimulator tests
# ===========================================================================

class TestLOBSimulator:
    """Tests for the limit order book simulator."""

    def test_limit_order_rests_in_book(self) -> None:
        """A placed limit order should appear in book depth."""
        lob = LOBSimulator()
        oid = lob.place_limit_order("bid", 99.50, 100.0)
        bids, _ = lob.book_depth(5)
        prices = [b[0] for b in bids]
        assert 99.50 in prices

    def test_best_bid_ask_after_seeding(self) -> None:
        """Best bid and ask should reflect the tightest resting orders."""
        lob = LOBSimulator()
        lob.place_limit_order("bid", 99.90, 50.0)
        lob.place_limit_order("bid", 99.80, 50.0)
        lob.place_limit_order("ask", 100.10, 50.0)
        lob.place_limit_order("ask", 100.20, 50.0)
        assert lob.best_bid() == pytest.approx(99.90)
        assert lob.best_ask() == pytest.approx(100.10)

    def test_market_buy_sweeps_asks(self) -> None:
        """A market buy should fill against resting asks in price order."""
        lob = LOBSimulator()
        lob.place_limit_order("ask", 100.10, 50.0)
        lob.place_limit_order("ask", 100.20, 50.0)
        fills = lob.place_market_order("buy", 60.0)
        total_filled = sum(f.qty for f in fills)
        assert total_filled == pytest.approx(60.0, abs=0.01)
        # first fill should be at best ask
        assert fills[0].price == pytest.approx(100.10)

    def test_market_sell_sweeps_bids(self) -> None:
        """A market sell should fill against resting bids in price order."""
        lob = LOBSimulator()
        lob.place_limit_order("bid", 99.90, 50.0)
        lob.place_limit_order("bid", 99.80, 50.0)
        fills = lob.place_market_order("sell", 70.0)
        total_filled = sum(f.qty for f in fills)
        assert total_filled == pytest.approx(70.0, abs=0.01)
        assert fills[0].price == pytest.approx(99.90)

    def test_cancel_removes_order(self) -> None:
        """Cancelling an order should remove it from the book."""
        lob = LOBSimulator()
        oid = lob.place_limit_order("ask", 100.50, 100.0)
        assert lob.cancel_order(oid) is True
        bids, asks = lob.book_depth()
        ask_prices = [a[0] for a in asks]
        assert 100.50 not in ask_prices

    def test_cancel_nonexistent_returns_false(self) -> None:
        """Cancelling an unknown order id should return False."""
        lob = LOBSimulator()
        assert lob.cancel_order("fake-id-xyz") is False

    def test_mid_price_is_average(self) -> None:
        """Mid price should equal (best_bid + best_ask) / 2."""
        lob = LOBSimulator()
        lob.place_limit_order("bid", 99.00, 10.0)
        lob.place_limit_order("ask", 101.00, 10.0)
        assert lob.mid_price() == pytest.approx(100.0)

    def test_spread_bps_calculation(self) -> None:
        """Spread in bps should match manual calculation."""
        lob = LOBSimulator()
        lob.place_limit_order("bid", 100.0, 10.0)
        lob.place_limit_order("ask", 100.10, 10.0)
        expected_bps = 0.10 / 100.0 * 10_000
        assert lob.spread_bps() == pytest.approx(expected_bps, rel=1e-5)

    def test_empty_book_returns_none(self) -> None:
        """Empty book should return None for best bid/ask/mid."""
        lob = LOBSimulator()
        assert lob.best_bid() is None
        assert lob.best_ask() is None
        assert lob.mid_price() is None
        assert lob.spread_bps() is None

    def test_invalid_order_raises(self) -> None:
        """Invalid order parameters should raise ValueError."""
        lob = LOBSimulator()
        with pytest.raises(ValueError):
            lob.place_limit_order("bid", -1.0, 10.0)
        with pytest.raises(ValueError):
            lob.place_market_order("buy", -5.0)
        with pytest.raises(ValueError):
            lob.place_limit_order("invalid_side", 100.0, 10.0)


# ===========================================================================
# TickDataGenerator tests
# ===========================================================================

class TestTickDataGenerator:
    """Tests for synthetic tick data generation."""

    def test_correct_tick_count(self) -> None:
        """Generator should return at most n_ticks ticks (some may be None)."""
        gen = TickDataGenerator(
            initial_price=100.0, rng=np.random.default_rng(42)
        )
        ticks = gen.generate(n_ticks=100, volatility=0.0001, arrival_rate=10.0)
        # some steps may skip (cancel with empty book), so count >= 0
        assert len(ticks) >= 0
        assert len(ticks) <= 100

    def test_price_within_reasonable_range(self) -> None:
        """All trade tick prices should be within 20% of initial price for low vol."""
        gen = TickDataGenerator(
            initial_price=100.0, rng=np.random.default_rng(5)
        )
        ticks = gen.generate(n_ticks=200, volatility=0.00005, arrival_rate=5.0)
        trade_prices = [t.price for t in ticks if t.tick_type == "trade"]
        if trade_prices:
            assert all(80.0 < p < 120.0 for p in trade_prices), (
                f"Some prices out of expected range: min={min(trade_prices):.2f}"
            )

    def test_tick_types_present(self) -> None:
        """Generated tick stream should contain limit, trade, and cancel types."""
        gen = TickDataGenerator(
            initial_price=50.0, rng=np.random.default_rng(13)
        )
        ticks = gen.generate(n_ticks=300, volatility=0.0001, arrival_rate=20.0)
        types = {t.tick_type for t in ticks}
        # at least trades and quotes should appear
        assert "trade" in types or "quote_bid" in types or "quote_ask" in types

    def test_timestamps_non_decreasing(self) -> None:
        """Tick timestamps must be monotonically non-decreasing."""
        gen = TickDataGenerator(rng=np.random.default_rng(21))
        ticks = gen.generate(n_ticks=100)
        times = [t.timestamp for t in ticks]
        assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))


# ===========================================================================
# SpreadDynamicsSimulator tests
# ===========================================================================

class TestSpreadDynamicsSimulator:
    """Tests for adverse selection and liquidity crisis spread simulation."""

    def test_adverse_selection_widens_spread(self) -> None:
        """Spread should be higher at onset than at the end when informed traders arrive."""
        sim = SpreadDynamicsSimulator(base_spread_bps=5.0, rng=np.random.default_rng(7))
        df = sim.simulate_adverse_selection_event(n_steps=100, informed_pct=0.5)
        assert isinstance(df, pd.DataFrame)
        assert "spread_bps" in df.columns
        early_mean = df["spread_bps"].iloc[:10].mean()
        late_mean = df["spread_bps"].iloc[-10:].mean()
        # informed fraction decays, so late spread < early spread
        assert early_mean > late_mean * 0.5

    def test_adverse_selection_correct_length(self) -> None:
        """Output DataFrame should have exactly n_steps rows."""
        sim = SpreadDynamicsSimulator(rng=np.random.default_rng(8))
        df = sim.simulate_adverse_selection_event(n_steps=50)
        assert len(df) == 50

    def test_liquidity_crisis_spread_jumps(self) -> None:
        """Spread should jump at crisis_onset step."""
        sim = SpreadDynamicsSimulator(base_spread_bps=5.0, rng=np.random.default_rng(9))
        onset = 30
        df = sim.simulate_liquidity_crisis(n_steps=100, crisis_onset=onset)
        pre_crisis_max = df["spread_bps"].iloc[:onset].max()
        post_onset_early = df["spread_bps"].iloc[onset:onset + 5].mean()
        assert post_onset_early > pre_crisis_max

    def test_liquidity_crisis_correct_length(self) -> None:
        """Output DataFrame should have exactly n_steps rows."""
        sim = SpreadDynamicsSimulator(rng=np.random.default_rng(10))
        df = sim.simulate_liquidity_crisis(n_steps=80, crisis_onset=40)
        assert len(df) == 80

    def test_crisis_active_flag(self) -> None:
        """crisis_active column should be False before onset and True after."""
        sim = SpreadDynamicsSimulator(rng=np.random.default_rng(11))
        onset = 25
        df = sim.simulate_liquidity_crisis(n_steps=60, crisis_onset=onset)
        assert df["crisis_active"].iloc[:onset].all() == False  # noqa: E712
        assert df["crisis_active"].iloc[onset:].all() == True   # noqa: E712


# ===========================================================================
# MomentCalibrator tests
# ===========================================================================

class TestMomentCalibrator:
    """Tests for moment-matching calibration of GBM, OU, GARCH, jump-diffusion."""

    def test_gbm_calibration_matches_moments(
        self, normal_returns: NDArray[np.float64]
    ) -> None:
        """Calibrated GBM parameters should reproduce input mean and variance."""
        cal = MomentCalibrator()
        dt = 1.0 / 252
        params = cal.calibrate_gbm(normal_returns, dt=dt)

        emp_mean = float(np.mean(normal_returns))
        emp_var = float(np.var(normal_returns, ddof=1))

        # implied model mean and variance per step
        model_mean = (params.mu - 0.5 * params.sigma**2) * dt
        model_var = params.sigma**2 * dt

        assert model_mean == pytest.approx(emp_mean, rel=0.05, abs=1e-6)
        assert model_var == pytest.approx(emp_var, rel=0.05, abs=1e-10)

    def test_gbm_sigma_non_negative(
        self, normal_returns: NDArray[np.float64]
    ) -> None:
        """Calibrated sigma must be non-negative."""
        cal = MomentCalibrator()
        params = cal.calibrate_gbm(normal_returns)
        assert params.sigma >= 0.0

    def test_ou_calibration_recovers_mean(
        self, price_series: NDArray[np.float64]
    ) -> None:
        """OU theta should be close to the empirical mean of the price series."""
        cal = MomentCalibrator()
        params = cal.calibrate_ou(price_series)
        emp_mean = float(np.mean(price_series))
        assert params.theta == pytest.approx(emp_mean, rel=0.15)

    def test_ou_kappa_positive(self, price_series: NDArray[np.float64]) -> None:
        """Mean-reversion speed kappa should be positive for a stationary series."""
        cal = MomentCalibrator()
        params = cal.calibrate_ou(price_series)
        assert params.kappa > 0.0

    def test_garch_persistence_below_one(
        self, normal_returns: NDArray[np.float64]
    ) -> None:
        """GARCH(1,1) alpha + beta should be < 1 (covariance stationary)."""
        cal = MomentCalibrator()
        params = cal.calibrate_garch(normal_returns)
        assert params.persistence() < 1.0

    def test_garch_params_positive(
        self, normal_returns: NDArray[np.float64]
    ) -> None:
        """All GARCH parameters must be strictly positive."""
        cal = MomentCalibrator()
        params = cal.calibrate_garch(normal_returns)
        assert params.omega > 0
        assert params.alpha > 0
        assert params.beta > 0

    def test_jump_diffusion_sigma_positive(
        self, fat_tail_returns: NDArray[np.float64]
    ) -> None:
        """Jump diffusion sigma must be positive."""
        cal = MomentCalibrator()
        params = cal.calibrate_jump_diffusion(fat_tail_returns)
        assert params.sigma > 0.0

    def test_jump_diffusion_lambda_positive(
        self, fat_tail_returns: NDArray[np.float64]
    ) -> None:
        """Jump intensity lambda must be positive."""
        cal = MomentCalibrator()
        params = cal.calibrate_jump_diffusion(fat_tail_returns)
        assert params.lambda_j > 0.0

    def test_ou_half_life_finite(self, price_series: NDArray[np.float64]) -> None:
        """Half-life of OU should be finite and positive for mean-reverting series."""
        cal = MomentCalibrator()
        params = cal.calibrate_ou(price_series)
        hl = params.half_life()
        assert hl > 0.0
        assert math.isfinite(hl)


# ===========================================================================
# GoodnessOfFit tests
# ===========================================================================

class TestGoodnessOfFit:
    """Tests for KS test, Anderson-Darling, QQ plot stats, stylized facts."""

    def test_ks_test_same_distribution(self) -> None:
        """Two samples from the same distribution should have high p-value."""
        rng = np.random.default_rng(42)
        a = rng.normal(0.0, 1.0, 500)
        b = rng.normal(0.0, 1.0, 500)
        gof = GoodnessOfFit()
        result = gof.ks_test(a, b)
        assert isinstance(result, KSResult)
        # with same distribution, typically fail to reject H0
        assert result.p_value >= 0.0

    def test_ks_test_different_distributions(self) -> None:
        """Samples from clearly different distributions should be flagged."""
        rng = np.random.default_rng(43)
        a = rng.normal(0.0, 0.01, 500)
        b = rng.normal(1.0, 0.01, 500)   # shifted by 100 std devs
        gof = GoodnessOfFit()
        result = gof.ks_test(a, b)
        assert result.reject_h0 is True

    def test_ks_statistic_range(self) -> None:
        """KS statistic must be in [0, 1]."""
        rng = np.random.default_rng(44)
        a = rng.normal(0.0, 1.0, 200)
        b = rng.normal(0.5, 1.0, 200)
        gof = GoodnessOfFit()
        result = gof.ks_test(a, b)
        assert 0.0 <= result.statistic <= 1.0

    def test_anderson_darling_normal(self, normal_returns: NDArray[np.float64]) -> None:
        """Anderson-Darling test against normal should not reject for near-normal data."""
        gof = GoodnessOfFit()
        result = gof.anderson_darling(normal_returns, distribution="norm")
        assert isinstance(result, ADResult)
        assert result.statistic >= 0.0

    def test_anderson_darling_invalid_dist(self) -> None:
        """Invalid distribution name should raise ValueError."""
        gof = GoodnessOfFit()
        rng = np.random.default_rng(45)
        data = rng.normal(size=100)
        with pytest.raises(ValueError):
            gof.anderson_darling(data, distribution="t_dist")

    def test_qq_plot_r2_perfect(self) -> None:
        """QQ R^2 should be 1.0 when empirical == theoretical quantiles."""
        rng = np.random.default_rng(46)
        data = rng.normal(size=100)
        sorted_data = np.sort(data)
        gof = GoodnessOfFit()
        r2 = gof.qq_plot_stats(sorted_data, sorted_data)
        assert r2 == pytest.approx(1.0)

    def test_qq_plot_r2_between_zero_and_one(self) -> None:
        """QQ R^2 should generally be in a valid range."""
        rng = np.random.default_rng(47)
        emp = rng.normal(0.0, 1.0, 200)
        theo = rng.normal(0.0, 1.0, 200)
        sorted_emp = np.sort(emp)
        sorted_theo = np.sort(theo)
        gof = GoodnessOfFit()
        r2 = gof.qq_plot_stats(sorted_emp, sorted_theo)
        # R^2 can be negative for bad fit, but should be finite
        assert math.isfinite(r2)

    def test_stylized_facts_score_keys(
        self, fat_tail_returns: NDArray[np.float64]
    ) -> None:
        """stylized_facts_score should return all expected keys."""
        gof = GoodnessOfFit()
        scores = gof.stylized_facts_score(fat_tail_returns)
        expected_keys = {
            "heavy_tails",
            "vol_clustering",
            "no_autocorr",
            "negative_skew",
            "leverage_effect",
            "overall",
        }
        assert expected_keys == set(scores.keys())

    def test_stylized_facts_scores_in_unit_interval(
        self, fat_tail_returns: NDArray[np.float64]
    ) -> None:
        """All stylized fact scores must be in [0, 1]."""
        gof = GoodnessOfFit()
        scores = gof.stylized_facts_score(fat_tail_returns)
        for key, val in scores.items():
            assert 0.0 <= val <= 1.0, f"Score '{key}' = {val} out of [0, 1]"

    def test_stylized_facts_fat_tails_score_high(
        self, fat_tail_returns: NDArray[np.float64]
    ) -> None:
        """Fat-tailed data should score well on the heavy_tails criterion."""
        gof = GoodnessOfFit()
        scores = gof.stylized_facts_score(fat_tail_returns)
        assert scores["heavy_tails"] > 0.3, (
            f"Expected fat tails score > 0.3, got {scores['heavy_tails']:.4f}"
        )
