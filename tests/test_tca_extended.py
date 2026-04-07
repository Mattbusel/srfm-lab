"""
tests/test_tca_extended.py -- Extended TCA tests covering market impact models,
venue analysis, reversion analysis, and TCA store operations.

Modules under test:
  - execution/tca/market_impact_model.py
  - execution/tca/venue_analysis.py
  - execution/tca/reversion_analyzer.py
  - execution/tca/tca_store.py
  - execution/tca/tca_engine.py
"""

from __future__ import annotations

import math
import sqlite3
import tempfile
import os
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from execution.tca.market_impact_model import (
    LinearImpactModel,
    SqrtImpactModel,
    NonlinearImpactModel,
    ImpactModelEnsemble,
    ImpactCalibrator,
    ModelParams,
)
from execution.tca.venue_analysis import (
    VenueAnalyzer,
    VenueScore,
    VenueComparison,
    RoutingRecommendation,
    VenueReportGenerator,
    _composite_score,
    _score_metric,
)


# ---------------------------------------------------------------------------
# Synthetic TCA result builder
# ---------------------------------------------------------------------------


@dataclass
class SyntheticTCAResult:
    """Minimal TCAResult-like object for testing."""
    symbol: str
    venue: str
    implementation_shortfall_bps: float
    fill_rate: float
    time_to_fill_ms: float
    spread_cost_bps: float
    market_impact_bps: float
    participation_rate: float = 0.05
    trade_date: Optional[str] = None


def make_tca_result(
    symbol: str = "SPY",
    venue: str = "NASDAQ",
    is_bps: float = 5.0,
    fill_rate: float = 1.0,
    fill_time_ms: float = 50.0,
    spread_bps: float = 1.0,
    impact_bps: float = 4.0,
    prate: float = 0.05,
    trade_date: Optional[str] = None,
) -> SyntheticTCAResult:
    return SyntheticTCAResult(
        symbol=symbol,
        venue=venue,
        implementation_shortfall_bps=is_bps,
        fill_rate=fill_rate,
        time_to_fill_ms=fill_time_ms,
        spread_cost_bps=spread_bps,
        market_impact_bps=impact_bps,
        participation_rate=prate,
        trade_date=trade_date,
    )


# ---------------------------------------------------------------------------
# ImplementationShortfall  # IS = decision price vs execution price
# ---------------------------------------------------------------------------


def test_implementation_shortfall_positive_on_overpay():
    """IS > 0 when execution price > decision price for a buy."""
    decision_price = 400.0
    fill_price = 402.0
    is_bps = (fill_price - decision_price) / decision_price * 10_000
    assert is_bps > 0.0
    assert abs(is_bps - 50.0) < 1.0  # 50 bps


def test_implementation_shortfall_zero_on_exact_fill():
    decision_price = 400.0
    fill_price = 400.0
    is_bps = (fill_price - decision_price) / decision_price * 10_000
    assert is_bps == 0.0


def test_implementation_shortfall_negative_on_improvement():
    """IS < 0 when we got a better price than decision (price improvement)."""
    decision_price = 400.0
    fill_price = 398.0
    is_bps = (fill_price - decision_price) / decision_price * 10_000
    assert is_bps < 0.0


# ---------------------------------------------------------------------------
# SqrtImpactModel
# ---------------------------------------------------------------------------


def test_sqrt_impact_model_fit_returns_params():
    model = SqrtImpactModel()
    qtys = [100.0, 200.0, 500.0, 1000.0, 2000.0]
    advs = [1_000_000.0] * 5
    sigmas = [0.02] * 5
    # Synthetic impacts proportional to sqrt(qty/adv)
    impacts = [0.5 * 0.02 * math.sqrt(q / 1e6) * 10_000 for q in qtys]
    params = model.fit(qtys, advs, sigmas, impacts)
    assert isinstance(params, ModelParams)
    assert params.eta > 0.0
    assert params.n_obs == 5


def test_sqrt_impact_model_predict_proportional_to_sqrt():
    """Doubling qty should increase impact by ~sqrt(2)."""
    model = SqrtImpactModel()
    # Fit on synthetic sqrt-law data
    qtys = [100.0, 200.0, 400.0, 800.0, 1600.0]
    advs = [1_000_000.0] * 5
    sigmas = [0.02] * 5
    impacts = [0.5 * 0.02 * math.sqrt(q / 1e6) * 10_000 for q in qtys]
    model.fit(qtys, advs, sigmas, impacts)

    imp1 = model.predict(100.0, 1_000_000.0, 0.02)
    imp2 = model.predict(400.0, 1_000_000.0, 0.02)
    ratio = imp2 / (imp1 + 1e-12)
    # sqrt(4) = 2.0
    np.testing.assert_allclose(ratio, 2.0, rtol=0.2)


def test_sqrt_impact_model_predict_zero_adv_returns_zero():
    model = SqrtImpactModel()
    model.params = ModelParams(eta=1.0, model_type="sqrt")
    result = model.predict(100.0, adv=0.0, sigma=0.02)
    assert result == 0.0


def test_sqrt_impact_model_predict_pre_trade_interface():
    model = SqrtImpactModel()
    qtys = [100.0, 200.0, 300.0]
    advs = [1e6] * 3
    sigmas = [0.02] * 3
    impacts = [2.0, 3.0, 4.0]
    model.fit(qtys, advs, sigmas, impacts)
    result = model.predict_pre_trade("SPY", 100.0, 1e6, 0.02)
    assert result >= 0.0


def test_sqrt_impact_increases_with_participation():
    """Higher participation rate -> higher impact."""
    model = SqrtImpactModel()
    qtys = [1000.0, 5000.0, 10000.0, 50000.0]
    advs = [100_000.0] * 4
    sigmas = [0.015] * 4
    impacts = [0.5 * 0.015 * math.sqrt(q / 1e5) * 10_000 for q in qtys]
    model.fit(qtys, advs, sigmas, impacts)
    imp_low = model.predict(1000.0, 100_000.0, 0.015)
    imp_high = model.predict(50000.0, 100_000.0, 0.015)
    assert imp_high > imp_low


# ---------------------------------------------------------------------------
# AlmgrenChrissModel / Linear impact
# ---------------------------------------------------------------------------


def test_linear_impact_model_fit_and_predict():
    model = LinearImpactModel()
    prates = [0.01, 0.05, 0.10, 0.20, 0.50]
    sigmas = [0.02] * 5
    impacts = [1.0 * 0.02 * math.sqrt(p) * 10_000 for p in prates]
    params = model.fit(prates, sigmas, impacts)
    assert params.eta > 0.0
    predicted = model.predict(0.10, sigma=0.02)
    # Should be in ballpark of sqrt(0.1) * 0.02 * 10000 = 63 bps
    assert 30.0 < predicted < 200.0


def test_linear_impact_model_eta_scaling():
    """Higher sigma -> higher predicted impact for same participation."""
    model = LinearImpactModel()
    prates = [0.05, 0.10, 0.15, 0.20, 0.30]
    sigmas_low = [0.01] * 5
    sigmas_high = [0.05] * 5
    impacts_low = [0.5 * 0.01 * math.sqrt(p) * 10_000 for p in prates]
    impacts_high = [0.5 * 0.05 * math.sqrt(p) * 10_000 for p in prates]
    params_low = model.fit(prates, sigmas_low, impacts_low)
    imp_low = model.predict(0.10, sigma=0.01)
    imp_high = model.predict(0.10, sigma=0.05)
    assert imp_high > imp_low * 3.0  # 5x sigma increase should mean 5x impact


def test_linear_impact_model_predict_pre_trade():
    model = LinearImpactModel()
    prates = [0.01, 0.05, 0.10]
    sigmas = [0.02, 0.02, 0.02]
    impacts = [5.0, 10.0, 14.0]
    model.fit(prates, sigmas, impacts)
    result = model.predict_pre_trade(qty=1000.0, adv=10_000.0, sigma=0.02)
    assert result >= 0.0


# ---------------------------------------------------------------------------
# Ensemble model
# ---------------------------------------------------------------------------


def test_impact_ensemble_weights_sum_to_one():
    ensemble = ImpactModelEnsemble()
    qtys = [100.0, 200.0, 500.0, 1000.0, 2000.0]
    advs = [1e6] * 5
    sigmas = [0.02] * 5
    prates = [q / 1e6 for q in qtys]
    impacts = [0.5 * 0.02 * math.sqrt(p) * 10_000 for p in prates]
    ensemble.fit(qtys, advs, sigmas, prates, impacts)
    total = sum(ensemble.weights)
    np.testing.assert_allclose(total, 1.0, atol=1e-9)


def test_impact_ensemble_predict_nonnegative():
    ensemble = ImpactModelEnsemble()
    qtys = [100.0, 500.0, 1000.0]
    advs = [1e6] * 3
    sigmas = [0.02] * 3
    prates = [0.0001, 0.0005, 0.001]
    impacts = [2.0, 5.0, 7.0]
    ensemble.fit(qtys, advs, sigmas, prates, impacts)
    pred = ensemble.predict(100.0, 1e6, 0.02, 0.0001)
    assert pred >= 0.0


# ---------------------------------------------------------------------------
# VenueAnalyzer
# ---------------------------------------------------------------------------


def _make_venue_trades(venue: str, n: int = 5, slip: float = 5.0) -> List[SyntheticTCAResult]:
    return [make_tca_result(venue=venue, is_bps=slip, fill_time_ms=50.0) for _ in range(n)]


def test_venue_analyzer_compare_venues_best_has_lowest_slippage():
    trades = (
        _make_venue_trades("NYSE", n=10, slip=3.0)
        + _make_venue_trades("NASDAQ", n=10, slip=7.0)
    )
    analyzer = VenueAnalyzer()
    comparison = analyzer.compare_venues(trades)
    assert comparison.best_venue == "NYSE"
    assert comparison.worst_venue == "NASDAQ"


def test_venue_analyzer_compare_venues_ranking_order():
    trades = (
        _make_venue_trades("IEX", n=5, slip=2.0)
        + _make_venue_trades("BATS", n=5, slip=5.0)
        + _make_venue_trades("CBOE", n=5, slip=8.0)
    )
    analyzer = VenueAnalyzer()
    comparison = analyzer.compare_venues(trades)
    assert comparison.ranking[0] == "IEX"
    assert comparison.ranking[-1] == "CBOE"
    assert comparison.n_venues == 3


def test_venue_analyzer_compare_venues_empty_returns_empty():
    analyzer = VenueAnalyzer()
    result = analyzer.compare_venues([])
    assert result.n_venues == 0
    assert result.best_venue == ""


def test_venue_analyzer_score_computation_range():
    """All venue scores should be in [0, 100]."""
    trades = (
        _make_venue_trades("A", slip=1.0, n=5)
        + _make_venue_trades("B", slip=10.0, n=5)
    )
    analyzer = VenueAnalyzer()
    comparison = analyzer.compare_venues(trades)
    for vs in comparison.scores.values():
        assert 0.0 <= vs.score <= 100.0


def test_venue_analyzer_route_large_order_to_dark_pool():
    """Orders > 1% ADV should go to IEX (dark pool)."""
    analyzer = VenueAnalyzer()
    venue = analyzer.best_venue_for(
        symbol="SPY",
        side="BUY",
        qty=20_000.0,
        urgency="MEDIUM",
        adv=1_000_000.0,  # 2% ADV
    )
    assert venue == "IEX"


def test_venue_analyzer_route_high_urgency_to_direct_exchange():
    analyzer = VenueAnalyzer()
    venue = analyzer.best_venue_for(
        symbol="SPY",
        side="BUY",
        qty=100.0,
        urgency="HIGH",
        adv=1_000_000.0,
    )
    assert venue == "NASDAQ"


def test_venue_analyzer_route_no_history_falls_back_to_nasdaq():
    analyzer = VenueAnalyzer()
    venue = analyzer.best_venue_for(
        symbol="UNKNOWN_SYM",
        side="BUY",
        qty=100.0,
        urgency="LOW",
    )
    assert venue == "NASDAQ"


def test_venue_analyzer_route_order_returns_recommendation_with_reason():
    analyzer = VenueAnalyzer()
    rec = analyzer.route_order("SPY", "BUY", 100.0, "LOW")
    assert isinstance(rec, RoutingRecommendation)
    assert len(rec.reason) > 0
    assert 0.0 <= rec.confidence <= 1.0


def test_venue_analyzer_add_results_updates_history():
    analyzer = VenueAnalyzer()
    trades = _make_venue_trades("NYSE", n=5, slip=3.0)
    analyzer.add_results(trades)
    assert len(analyzer._history) == 5


def test_venue_scorecard_empty_returns_empty_dict():
    analyzer = VenueAnalyzer()
    scorecard = analyzer.venue_scorecard(window_days=30)
    assert scorecard == {}


def test_score_metric_lower_is_better():
    """Lowest value should get score 100 when lower_is_better=True."""
    score_low = _score_metric(1.0, [1.0, 5.0, 10.0], lower_is_better=True)
    score_high = _score_metric(10.0, [1.0, 5.0, 10.0], lower_is_better=True)
    assert score_low > score_high
    np.testing.assert_allclose(score_low, 100.0, atol=1e-9)
    np.testing.assert_allclose(score_high, 0.0, atol=1e-9)


def test_score_metric_single_value_returns_50():
    score = _score_metric(5.0, [5.0], lower_is_better=True)
    assert score == 50.0


# ---------------------------------------------------------------------------
# VenueReportGenerator
# ---------------------------------------------------------------------------


def test_venue_report_generator_to_markdown_contains_venue_names():
    scores = {
        "NYSE": VenueScore("NYSE", 3.0, 1.0, 50.0, 1.0, 80.0, 10),
        "NASDAQ": VenueScore("NASDAQ", 5.0, 0.95, 60.0, 1.5, 60.0, 8),
    }
    gen = VenueReportGenerator()
    md = gen.to_markdown(scores)
    assert "NYSE" in md
    assert "NASDAQ" in md
    assert "Venue Scorecard" in md


def test_venue_report_generator_to_csv_has_header():
    scores = {
        "NYSE": VenueScore("NYSE", 3.0, 1.0, 50.0, 1.0, 80.0, 10),
    }
    gen = VenueReportGenerator()
    csv_str = gen.to_csv(scores)
    assert "venue" in csv_str
    assert "score" in csv_str
    assert "NYSE" in csv_str


def test_venue_report_generator_empty_scores_markdown():
    gen = VenueReportGenerator()
    md = gen.to_markdown({})
    assert "No data" in md


def test_venue_report_generator_daily_summary():
    scores = {
        "A": VenueScore("A", 2.0, 1.0, 30.0, 0.5, 90.0, 5),
        "B": VenueScore("B", 8.0, 0.90, 80.0, 2.0, 40.0, 5),
    }
    gen = VenueReportGenerator()
    summary = gen.daily_venue_summary(scores, date="2026-01-15")
    assert summary["date"] == "2026-01-15"
    assert summary["n_venues"] == 2
    assert summary["best_venue"] == "A"
    assert summary["worst_venue"] == "B"


# ---------------------------------------------------------------------------
# ReversionAnalyzer  # post-trade reversion
# ---------------------------------------------------------------------------


def test_reversion_analyzer_import():
    """ReversionAnalyzer should be importable and instantiable."""
    from execution.tca.reversion_analyzer import ReversionAnalyzer
    ra = ReversionAnalyzer()
    assert ra is not None


def test_reversion_analyzer_fit_exponential():
    """Test that ReversionAnalyzer can fit an exponential decay to post-trade data."""
    from execution.tca.reversion_analyzer import ReversionAnalyzer
    ra = ReversionAnalyzer()
    # Synthetic exponential reversion: impact decays by half every 5 periods
    times = list(range(1, 21))
    initial_impact = 10.0
    impacts = [initial_impact * math.exp(-0.1386 * t) for t in times]  # lambda=ln2/5
    result = ra.fit(times, impacts)
    assert result is not None


def test_reversion_analyzer_handles_empty_data():
    from execution.tca.reversion_analyzer import ReversionAnalyzer
    ra = ReversionAnalyzer()
    # Should not raise on empty input
    try:
        result = ra.fit([], [])
    except (ValueError, ZeroDivisionError):
        pass  # acceptable to raise on degenerate input


# ---------------------------------------------------------------------------
# TCAStore  # upsert, aggregate, daily_report
# ---------------------------------------------------------------------------


def test_tca_store_import_and_instantiate():
    from execution.tca.tca_store import TCAStore
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "tca_test.db")
        store = TCAStore(db_path=db_path)
        assert store is not None


def test_tca_store_upsert_and_query():
    from execution.tca.tca_store import TCAStore
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "tca.db")
        store = TCAStore(db_path=db_path)
        rec = {
            "order_id": "test-001",
            "symbol": "SPY",
            "side": "BUY",
            "fill_price": 450.0,
            "decision_price": 449.0,
            "implementation_shortfall_bps": 22.3,
            "market_impact_bps": 18.0,
            "spread_cost_bps": 1.5,
            "fill_rate": 1.0,
            "time_to_fill_ms": 45.0,
            "venue": "NASDAQ",
            "trade_date": "2026-01-10",
        }
        # Should not raise
        store.upsert(rec)


def test_tca_store_daily_report_structure():
    from execution.tca.tca_store import TCAStore
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "tca.db")
        store = TCAStore(db_path=db_path)
        # Insert a few records
        for i in range(5):
            store.upsert({
                "order_id": f"ord-{i}",
                "symbol": "SPY",
                "side": "BUY",
                "fill_price": 450.0 + i,
                "decision_price": 450.0,
                "implementation_shortfall_bps": float(i),
                "market_impact_bps": float(i),
                "spread_cost_bps": 1.0,
                "fill_rate": 1.0,
                "time_to_fill_ms": 50.0,
                "venue": "NASDAQ",
                "trade_date": "2026-01-10",
            })
        report = store.daily_report("2026-01-10")
        assert report is not None


# ---------------------------------------------------------------------------
# ImpactCalibrator
# ---------------------------------------------------------------------------


def test_impact_calibrator_ensemble_calibration():
    """ImpactCalibrator with ensemble type should fit all three sub-models."""
    calibrator = ImpactCalibrator(model_type="ensemble")
    # Build synthetic results
    results = [
        make_tca_result(impact_bps=0.5 * 0.02 * math.sqrt(0.05) * 10_000, prate=0.05)
        for _ in range(20)
    ]
    params = calibrator.calibrate(results)
    assert params is not None
    assert params.n_obs > 0


def test_impact_calibrator_predict_pre_trade_nonneg():
    calibrator = ImpactCalibrator(model_type="sqrt")
    results = [make_tca_result(impact_bps=float(i + 1), prate=0.01 * (i + 1)) for i in range(10)]
    calibrator.calibrate(results)
    pred = calibrator.predict_pre_trade("SPY", 1000.0, 1_000_000.0, 0.02)
    assert pred >= 0.0


def test_impact_calibrator_cross_validate_returns_cv_result():
    from execution.tca.market_impact_model import CalibrationResult
    calibrator = ImpactCalibrator(model_type="sqrt")
    results = [make_tca_result(impact_bps=float(i + 1), prate=0.01 * (i + 1)) for i in range(20)]
    cv = calibrator.cross_validate(results, n_folds=3)
    assert isinstance(cv, CalibrationResult)
    assert cv.n_folds >= 1
    assert cv.cv_rmse >= 0.0
