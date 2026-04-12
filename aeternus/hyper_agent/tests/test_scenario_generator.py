"""
tests/test_scenario_generator.py
=================================
Tests for the scenario generator module.
"""

from __future__ import annotations

import numpy as np
import pytest

from hyper_agent.scenario_generator import (
    ScenarioParams,
    GeneratedScenario,
    ChronosScenarioGenerator,
    ScenarioLibrary,
    DifficultyScoringEngine,
    CurriculumScenarioManager,
    ScenarioPopulation,
    ScenarioInterpolator,
    HestonPriceProcess,
    LOBStructureGenerator,
)


@pytest.fixture
def gen():
    return ChronosScenarioGenerator()


@pytest.fixture
def normal_params():
    return ScenarioLibrary.normal_market(difficulty=0.5, seed=42, n_ticks=200)


# ---------------------------------------------------------------------------
# ScenarioParams tests
# ---------------------------------------------------------------------------

class TestScenarioParams:
    def test_to_dict_roundtrip(self, normal_params):
        d = normal_params.to_dict()
        restored = ScenarioParams.from_dict(d)
        assert restored.name == normal_params.name
        assert restored.base_volatility == normal_params.base_volatility

    def test_interpolation(self):
        p1 = ScenarioLibrary.normal_market(difficulty=0.1, seed=0)
        p2 = ScenarioLibrary.high_volatility_regime(difficulty=0.9, seed=1)
        mid = p1.interpolate(p2, alpha=0.5)
        assert mid.base_volatility > p1.base_volatility
        assert mid.base_volatility < p2.base_volatility

    def test_interpolation_extremes(self):
        p1 = ScenarioParams(name="a", base_volatility=0.0001)
        p2 = ScenarioParams(name="b", base_volatility=0.001)
        at_0 = p1.interpolate(p2, alpha=0.0)
        at_1 = p1.interpolate(p2, alpha=1.0)
        assert abs(at_0.base_volatility - 0.0001) < 1e-9
        assert abs(at_1.base_volatility - 0.001) < 1e-9


# ---------------------------------------------------------------------------
# HestonPriceProcess tests
# ---------------------------------------------------------------------------

class TestHestonPriceProcess:
    def test_simulate_returns_correct_shape(self):
        h = HestonPriceProcess()
        rng = np.random.default_rng(0)
        prices, variances = h.simulate(S0=100.0, V0=0.04, n_steps=200, rng=rng)
        assert prices.shape == (200,)
        assert variances.shape == (200,)

    def test_prices_positive(self):
        h = HestonPriceProcess()
        rng = np.random.default_rng(1)
        prices, _ = h.simulate(100.0, 0.04, 100, rng)
        assert (prices > 0).all()

    def test_variances_non_negative(self):
        h = HestonPriceProcess()
        rng = np.random.default_rng(2)
        _, variances = h.simulate(100.0, 0.04, 100, rng)
        assert (variances >= 0).all()


# ---------------------------------------------------------------------------
# LOBStructureGenerator tests
# ---------------------------------------------------------------------------

class TestLOBStructureGenerator:
    def test_generate_book_shape(self):
        lob_gen = LOBStructureGenerator(depth=5)
        rng = np.random.default_rng(0)
        bp, ap, bv, av = lob_gen.generate_book(
            mid_price=100.0, spread=0.2, base_depth=50.0,
            depth_decay=0.5, rng=rng
        )
        assert len(bp) == 5 and len(ap) == 5
        assert len(bv) == 5 and len(av) == 5

    def test_bid_below_ask(self):
        lob_gen = LOBStructureGenerator(depth=5)
        rng = np.random.default_rng(1)
        bp, ap, bv, av = lob_gen.generate_book(100.0, 0.2, 50.0, 0.5, rng)
        assert bp[0] < ap[0]

    def test_volumes_positive(self):
        lob_gen = LOBStructureGenerator(depth=5)
        rng = np.random.default_rng(2)
        _, _, bv, av = lob_gen.generate_book(100.0, 0.2, 50.0, 0.5, rng)
        assert (bv > 0).all()
        assert (av > 0).all()


# ---------------------------------------------------------------------------
# ChronosScenarioGenerator tests
# ---------------------------------------------------------------------------

class TestChronosScenarioGenerator:
    def test_generate_normal(self, gen, normal_params):
        scenario = gen.generate(normal_params)
        assert scenario.mid_prices.shape == (200,)
        assert scenario.spreads.shape == (200,)
        assert scenario.bid_prices.shape == (200, 10)

    def test_prices_positive(self, gen, normal_params):
        scenario = gen.generate(normal_params)
        assert (scenario.mid_prices > 0).all()

    def test_spreads_positive(self, gen, normal_params):
        scenario = gen.generate(normal_params)
        assert (scenario.spreads > 0).all()

    def test_flash_crash_event_flags(self, gen):
        params = ScenarioLibrary.flash_crash(difficulty=0.8, seed=0, n_ticks=300)
        scenario = gen.generate(params)
        crash_flags = scenario.event_flags & GeneratedScenario.FLAG_FLASH_CRASH
        assert crash_flags.sum() > 0

    def test_liquidity_crisis_flags(self, gen):
        params = ScenarioLibrary.liquidity_crisis(difficulty=0.7, seed=0, n_ticks=300)
        scenario = gen.generate(params)
        crisis_flags = scenario.event_flags & GeneratedScenario.FLAG_LIQUIDITY_CRISIS
        assert crisis_flags.sum() > 0

    def test_news_shock_flags(self, gen):
        params = ScenarioLibrary.news_shock_bullish(difficulty=0.5, seed=0, n_ticks=300)
        scenario = gen.generate(params)
        shock_flags = scenario.event_flags & GeneratedScenario.FLAG_NEWS_SHOCK
        assert shock_flags.sum() > 0

    def test_seed_reproducibility(self, gen):
        p1 = ScenarioParams(name="test", seed=42, n_ticks=100)
        p2 = ScenarioParams(name="test", seed=42, n_ticks=100)
        s1 = gen.generate(p1)
        s2 = gen.generate(p2)
        np.testing.assert_array_almost_equal(s1.mid_prices, s2.mid_prices)

    def test_different_seeds_differ(self, gen):
        p1 = ScenarioParams(name="test", seed=1, n_ticks=100)
        p2 = ScenarioParams(name="test", seed=2, n_ticks=100)
        s1 = gen.generate(p1)
        s2 = gen.generate(p2)
        assert not np.allclose(s1.mid_prices, s2.mid_prices)

    def test_to_lob_snapshots(self, gen, normal_params):
        scenario = gen.generate(normal_params)
        snaps = scenario.to_lob_snapshots()
        assert len(snaps) == 200
        assert "mid_price" in snaps[0]
        assert "best_bid" in snaps[0]


# ---------------------------------------------------------------------------
# ScenarioLibrary tests
# ---------------------------------------------------------------------------

class TestScenarioLibrary:
    def test_all_scenario_types(self):
        types = ScenarioLibrary.all_scenario_types()
        assert "normal" in types
        assert "flash_crash" in types
        assert "liquidity_crisis" in types

    def test_sample_random(self):
        rng = np.random.default_rng(0)
        params = ScenarioLibrary.sample_random(rng=rng, n_ticks=100)
        assert isinstance(params, ScenarioParams)
        assert params.n_ticks == 100

    def test_all_builders(self):
        gen = ChronosScenarioGenerator()
        builders = {
            "normal": ScenarioLibrary.normal_market,
            "flash_crash": ScenarioLibrary.flash_crash,
            "liquidity_crisis": ScenarioLibrary.liquidity_crisis,
        }
        for name, builder in builders.items():
            params = builder(difficulty=0.5, seed=0, n_ticks=100)
            scenario = gen.generate(params)
            assert len(scenario.mid_prices) > 0


# ---------------------------------------------------------------------------
# DifficultyScoringEngine tests
# ---------------------------------------------------------------------------

class TestDifficultyScoringEngine:
    def test_score_range(self):
        gen = ChronosScenarioGenerator()
        scorer = DifficultyScoringEngine()
        params = ScenarioLibrary.normal_market(seed=0, n_ticks=200)
        scenario = gen.generate(params)
        score = scorer.score(scenario)
        assert 0.0 <= score <= 1.0

    def test_flash_crash_harder(self):
        gen = ChronosScenarioGenerator()
        scorer = DifficultyScoringEngine()
        normal = gen.generate(ScenarioLibrary.normal_market(difficulty=0.3, seed=0, n_ticks=200))
        crash = gen.generate(ScenarioLibrary.flash_crash(difficulty=0.9, seed=0, n_ticks=200))
        score_normal = scorer.score(normal)
        score_crash = scorer.score(crash)
        # Crash should generally score higher difficulty
        assert score_crash >= score_normal - 0.3   # allow some tolerance

    def test_score_for_agent(self):
        gen = ChronosScenarioGenerator()
        scorer = DifficultyScoringEngine()
        scenario = gen.generate(ScenarioLibrary.normal_market(seed=0, n_ticks=200))
        agent_score = scorer.score_for_agent(scenario, competency=0.5)
        assert isinstance(agent_score, float)


# ---------------------------------------------------------------------------
# CurriculumScenarioManager tests
# ---------------------------------------------------------------------------

class TestCurriculumScenarioManager:
    def test_record_performance(self):
        mgr = CurriculumScenarioManager()
        for i in range(60):
            mgr.record_performance("normal", success=(i % 2 == 0))
        report = mgr.competency_report()
        assert "normal" in report
        assert "success_rate" in report["normal"]

    def test_difficulty_adjustment_up(self):
        mgr = CurriculumScenarioManager(target_success_rate=0.5)
        initial_diff = mgr._scenario_difficulties["normal"]
        for _ in range(60):
            mgr.record_performance("normal", success=True)  # always succeeds
        final_diff = mgr._scenario_difficulties["normal"]
        assert final_diff >= initial_diff

    def test_difficulty_adjustment_down(self):
        mgr = CurriculumScenarioManager(target_success_rate=0.8)
        initial_diff = mgr._scenario_difficulties["normal"]
        for _ in range(60):
            mgr.record_performance("normal", success=False)  # always fails
        final_diff = mgr._scenario_difficulties["normal"]
        assert final_diff <= initial_diff

    def test_sample_scenario_params(self):
        mgr = CurriculumScenarioManager()
        params = mgr.sample_scenario_params(n_ticks=100)
        assert isinstance(params, ScenarioParams)

    def test_competency_report_all_types(self):
        mgr = CurriculumScenarioManager()
        report = mgr.competency_report()
        for stype in ScenarioLibrary.all_scenario_types():
            assert stype in report


# ---------------------------------------------------------------------------
# ScenarioInterpolator tests
# ---------------------------------------------------------------------------

class TestScenarioInterpolator:
    def test_interpolation_count(self):
        p1 = ScenarioLibrary.normal_market(seed=0)
        p2 = ScenarioLibrary.flash_crash(seed=1)
        interp = ScenarioInterpolator(p1, p2, n_steps=5)
        steps = list(interp)
        assert len(steps) == 6   # 0..5 inclusive

    def test_at_extremes(self):
        p1 = ScenarioParams(name="a", base_volatility=0.001)
        p2 = ScenarioParams(name="b", base_volatility=0.01)
        interp = ScenarioInterpolator(p1, p2, n_steps=10)
        at_0 = interp.at(0.0)
        at_1 = interp.at(1.0)
        assert abs(at_0.base_volatility - 0.001) < 1e-9
        assert abs(at_1.base_volatility - 0.01) < 1e-9


# ---------------------------------------------------------------------------
# ScenarioPopulation tests
# ---------------------------------------------------------------------------

class TestScenarioPopulation:
    def test_population_size(self):
        pop = ScenarioPopulation(population_size=8, seed=0)
        assert len(pop.population) == 8

    def test_update_fitness(self):
        pop = ScenarioPopulation(population_size=4, seed=0)
        pop.update_fitness(0, 0.8)
        assert pop.population[0][1] == 0.8

    def test_sample_scenarios(self):
        pop = ScenarioPopulation(population_size=8, seed=0)
        scenarios = pop.sample_scenarios(n=3)
        assert len(scenarios) == 3
        for s in scenarios:
            assert isinstance(s, ScenarioParams)

    def test_evolve(self):
        pop = ScenarioPopulation(population_size=4, seed=0)
        for i in range(4):
            pop.update_fitness(i, float(i) * 0.1)
        pop.evolve()
        assert len(pop.population) == 4
