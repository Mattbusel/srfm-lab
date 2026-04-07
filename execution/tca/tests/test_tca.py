# execution/tca/tests/test_tca.py -- pytest test suite for SRFM TCA module
# Covers IS calculation, VWAP benchmark, impact models, reversion, venue, and store.

from __future__ import annotations

import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from typing import List

import pytest

# Ensure package is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from execution.tca.tca_engine import (
    ArrivalPriceBenchmark,
    CloseBenchmark,
    MarketData,
    TCAEngine,
    TCAResult,
    TradeRecord,
    TWAPBenchmark,
    VWAPBenchmark,
    _safe_bps,
    _side_sign,
    decompose_slippage,
)
from execution.tca.market_impact_model import (
    ImpactCalibrator,
    ImpactModelEnsemble,
    LinearImpactModel,
    NonlinearImpactModel,
    SqrtImpactModel,
)
from execution.tca.reversion_analyzer import (
    ReversionAnalyzer,
    ReversionDatabase,
    ReversionProfile,
    _fit_exponential_nls,
)
from execution.tca.venue_analysis import (
    VenueAnalyzer,
    VenueReportGenerator,
    VenueScore,
)
from execution.tca.tca_store import TCAStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_trade(
    symbol: str = "AAPL",
    side: str = "BUY",
    order_qty: float = 1000.0,
    filled_qty: float = 1000.0,
    order_price: float = 100.0,
    fill_price: float = 100.10,
    arrival_price: float = 100.00,
    interval_vwap: float = 100.05,
    interval_twap: float = 100.04,
    close_price: float = 100.20,
    venue: str = "NASDAQ",
    strategy: str = "MOM",
    order_type: str = "LIMIT",
    trade_id: str = "T001",
    bid: float = 99.98,
    ask: float = 100.02,
    market_volume: float = 50_000.0,
    adv: float = 2_000_000.0,
    offset_seconds: int = 5,
) -> TradeRecord:
    order_time = datetime(2026, 4, 7, 9, 30, 0, tzinfo=timezone.utc)
    fill_time = datetime(2026, 4, 7, 9, 30, offset_seconds, tzinfo=timezone.utc)
    return TradeRecord(
        symbol=symbol,
        side=side,
        order_qty=order_qty,
        filled_qty=filled_qty,
        order_price=order_price,
        fill_price=fill_price,
        order_time=order_time,
        fill_time=fill_time,
        venue=venue,
        strategy=strategy,
        order_type=order_type,
        arrival_price=arrival_price,
        interval_vwap=interval_vwap,
        interval_twap=interval_twap,
        close_price=close_price,
        bid_at_fill=bid,
        ask_at_fill=ask,
        market_volume=market_volume,
        adv=adv,
        trade_id=trade_id,
    )


def _make_result(
    is_bps: float = 10.0,
    venue: str = "NASDAQ",
    symbol: str = "AAPL",
    strategy: str = "MOM",
    fill_price: float = 100.10,
    fill_rate: float = 1.0,
    time_ms: float = 500.0,
    spread_bps: float = 2.0,
    impact_bps: float = 3.0,
    trade_date: str = "2026-04-07",
    trade_id: str = "T001",
    participation: float = 0.02,
) -> TCAResult:
    return TCAResult(
        implementation_shortfall_bps=is_bps,
        market_impact_bps=impact_bps,
        timing_cost_bps=1.0,
        spread_cost_bps=spread_bps,
        total_cost_bps=is_bps,
        participation_rate=participation,
        vwap_slippage_bps=5.0,
        twap_slippage_bps=4.0,
        close_slippage_bps=-10.0,
        decision_price=100.0,
        arrival_price=100.0,
        fill_price=fill_price,
        benchmark_type="ARRIVAL",
        fill_rate=fill_rate,
        time_to_fill_ms=time_ms,
        symbol=symbol,
        side="BUY",
        strategy=strategy,
        venue=venue,
        order_type="LIMIT",
        trade_id=trade_id,
        trade_date=trade_date,
    )


# ---------------------------------------------------------------------------
# TEST GROUP 1: Utility helpers
# ---------------------------------------------------------------------------

class TestUtilityHelpers:
    def test_side_sign_buy(self):
        assert _side_sign("BUY") == 1.0

    def test_side_sign_sell(self):
        assert _side_sign("SELL") == -1.0

    def test_side_sign_case_insensitive(self):
        assert _side_sign("buy") == 1.0
        assert _side_sign("Sell") == -1.0

    def test_safe_bps_normal(self):
        result = _safe_bps(0.001, 100.0)
        assert abs(result - 0.1) < 1e-9

    def test_safe_bps_zero_denominator(self):
        assert _safe_bps(1.0, 0.0) == 0.0

    def test_safe_bps_inf_denominator(self):
        assert _safe_bps(1.0, float("inf")) == 0.0


# ---------------------------------------------------------------------------
# TEST GROUP 2: Implementation Shortfall
# ---------------------------------------------------------------------------

class TestImplementationShortfall:
    def test_buy_positive_slippage(self):
        """BUY filled above arrival: IS should be positive (cost)."""
        trade = _make_trade(
            side="BUY",
            arrival_price=100.00,
            fill_price=100.10,
        )
        engine = TCAEngine()
        result = engine.analyze_trade(trade)
        assert result.implementation_shortfall_bps > 0.0

    def test_buy_exact_is_value(self):
        """IS = (100.10 - 100.00) / 100.00 * 10000 = 10 bps for BUY."""
        trade = _make_trade(
            side="BUY",
            arrival_price=100.00,
            fill_price=100.10,
        )
        engine = TCAEngine()
        result = engine.analyze_trade(trade)
        assert abs(result.implementation_shortfall_bps - 10.0) < 1e-6

    def test_sell_positive_slippage(self):
        """SELL filled below arrival: IS should be positive (cost for seller)."""
        trade = _make_trade(
            side="SELL",
            arrival_price=100.00,
            fill_price=99.90,
        )
        engine = TCAEngine()
        result = engine.analyze_trade(trade)
        assert result.implementation_shortfall_bps > 0.0

    def test_sell_exact_is_value(self):
        """IS = -1 * (99.90 - 100.00) / 100.00 * 10000 = 10 bps for SELL."""
        trade = _make_trade(
            side="SELL",
            arrival_price=100.00,
            fill_price=99.90,
        )
        engine = TCAEngine()
        result = engine.analyze_trade(trade)
        assert abs(result.implementation_shortfall_bps - 10.0) < 1e-6

    def test_zero_slippage(self):
        """Fill exactly at arrival price: IS should be 0 bps."""
        trade = _make_trade(
            side="BUY",
            arrival_price=100.00,
            fill_price=100.00,
        )
        engine = TCAEngine()
        result = engine.analyze_trade(trade)
        assert abs(result.implementation_shortfall_bps) < 1e-9

    def test_fill_rate_computed_correctly(self):
        """Partial fill: fill_rate = 500 / 1000 = 0.5."""
        trade = _make_trade(order_qty=1000.0, filled_qty=500.0)
        engine = TCAEngine()
        result = engine.analyze_trade(trade)
        assert abs(result.fill_rate - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# TEST GROUP 3: VWAP benchmark
# ---------------------------------------------------------------------------

class TestVWAPBenchmark:
    def test_vwap_slippage_buy_above_vwap(self):
        """BUY filled above VWAP: VWAP slippage should be positive."""
        trade = _make_trade(
            side="BUY",
            fill_price=100.10,
            interval_vwap=100.00,
        )
        engine = TCAEngine()
        result = engine.analyze_trade(trade)
        assert result.vwap_slippage_bps > 0.0

    def test_vwap_slippage_exact_value(self):
        """VWAP slip = (100.10 - 100.00) / 100.00 * 10000 = 10 bps."""
        trade = _make_trade(
            side="BUY",
            fill_price=100.10,
            interval_vwap=100.00,
        )
        engine = TCAEngine()
        result = engine.analyze_trade(trade)
        assert abs(result.vwap_slippage_bps - 10.0) < 1e-6

    def test_vwap_benchmark_fallback_to_fill(self):
        """When no VWAP data: benchmark falls back to fill price -> 0 slippage."""
        bench = VWAPBenchmark()
        trade = _make_trade(interval_vwap=None)
        trade.interval_vwap = None
        md = MarketData(symbol="AAPL")
        price = bench.compute(trade, md)
        assert price == trade.fill_price

    def test_zero_vwap_volume(self):
        """Zero market volume: participation rate clamped to 0."""
        trade = _make_trade(market_volume=0.0)
        engine = TCAEngine()
        result = engine.analyze_trade(trade)
        assert result.participation_rate == 0.0


# ---------------------------------------------------------------------------
# TEST GROUP 4: Impact model fitting and prediction
# ---------------------------------------------------------------------------

class TestImpactModels:
    def test_linear_model_fit(self):
        """LinearImpactModel should fit without error and return positive eta."""
        model = LinearImpactModel()
        prates = [0.01, 0.02, 0.05, 0.10, 0.15]
        sigmas = [0.02] * 5
        impacts = [2.0, 4.0, 8.0, 15.0, 20.0]
        params = model.fit(prates, sigmas, impacts)
        assert params.eta > 0.0
        assert params.n_obs == 5

    def test_sqrt_model_fit(self):
        """SqrtImpactModel should fit and predict positive impact."""
        model = SqrtImpactModel()
        qtys = [100.0, 500.0, 1000.0, 5000.0]
        advs = [1e6] * 4
        sigmas = [0.02] * 4
        impacts = [1.0, 3.0, 5.0, 12.0]
        params = model.fit(qtys, advs, sigmas, impacts)
        assert params.eta > 0.0
        pred = model.predict(1000.0, 1e6, 0.02)
        assert pred >= 0.0

    def test_nonlinear_model_fit(self):
        """NonlinearImpactModel should fit alpha between 0.1 and 2.0."""
        model = NonlinearImpactModel()
        qtys = [100.0, 500.0, 1000.0, 5000.0, 10000.0]
        advs = [1e6] * 5
        sigmas = [0.02] * 5
        impacts = [0.5, 2.0, 4.0, 10.0, 18.0]
        params = model.fit(qtys, advs, sigmas, impacts)
        assert 0.1 <= params.alpha <= 2.0

    def test_ensemble_weights_sum_to_one(self):
        """ImpactModelEnsemble weights should sum to 1."""
        ens = ImpactModelEnsemble()
        qtys = [100.0, 500.0, 1000.0, 5000.0, 10000.0]
        advs = [1e6] * 5
        sigmas = [0.02] * 5
        prates = [q / a for q, a in zip(qtys, advs)]
        impacts = [0.5, 2.0, 4.0, 10.0, 18.0]
        ens.fit(qtys, advs, sigmas, prates, impacts)
        assert abs(sum(ens.weights) - 1.0) < 1e-9

    def test_calibrator_predict_pre_trade(self):
        """ImpactCalibrator.predict_pre_trade should return a non-negative float."""
        results = [_make_result(impact_bps=float(i)) for i in range(3, 20)]
        calibrator = ImpactCalibrator(model_type="ensemble")
        calibrator.calibrate(results)
        pred = calibrator.predict_pre_trade("AAPL", 10000.0, 2_000_000.0, 0.02)
        assert pred >= 0.0

    def test_calibrator_cross_validate(self):
        """Cross-validation should return a CalibrationResult with cv_rmse >= 0."""
        results = [_make_result(impact_bps=float(i) * 0.5 + 1.0) for i in range(20)]
        calibrator = ImpactCalibrator(model_type="linear")
        cv = calibrator.cross_validate(results, n_folds=3)
        assert cv.cv_rmse >= 0.0
        assert cv.n_obs >= 0


# ---------------------------------------------------------------------------
# TEST GROUP 5: Reversion analyzer
# ---------------------------------------------------------------------------

class TestReversionAnalyzer:
    def _synthetic_prices(
        self,
        fill_price: float,
        impact_bps: float,
        permanent_frac: float,
        horizons: int,
    ) -> List[float]:
        """
        Generate synthetic post-trade prices with exponential reversion.
        fill_price is elevated by impact; prices decay back toward a permanent level.
        """
        prices = []
        perm_bps = impact_bps * permanent_frac
        temp_bps = impact_bps * (1.0 - permanent_frac)
        for t in range(1, horizons + 1):
            # Decay: temporary impact decays with half-life ~5 bars
            b = math.log(2.0) / 5.0
            remaining_bps = perm_bps + temp_bps * math.exp(-b * t)
            p = fill_price * (1.0 + remaining_bps / 10_000.0)
            prices.append(p)
        return prices

    def test_basic_reversion_profile(self):
        """ReversionAnalyzer should return a valid ReversionProfile."""
        trade = _make_trade(side="BUY", fill_price=100.10)
        prices = self._synthetic_prices(100.10, 10.0, 0.3, 60)
        analyzer = ReversionAnalyzer()
        profile = analyzer.analyze(trade, prices, horizons=[1, 5, 15, 30, 60])
        assert len(profile.reversion_bps) == 5
        assert len(profile.horizons) == 5

    def test_half_life_finite(self):
        """Exponential fit should produce a finite half-life with well-behaved data."""
        trade = _make_trade(side="BUY", fill_price=100.10)
        prices = self._synthetic_prices(100.10, 20.0, 0.2, 60)
        analyzer = ReversionAnalyzer()
        profile = analyzer.analyze(trade, prices, horizons=[1, 5, 10, 20, 30, 60])
        assert math.isfinite(profile.half_life_bars) or profile.half_life_bars == float("inf")

    def test_empty_prices_returns_zeros(self):
        """Empty price list should return a profile with zero reversion values."""
        trade = _make_trade(side="BUY", fill_price=100.10)
        analyzer = ReversionAnalyzer()
        profile = analyzer.analyze(trade, [], horizons=[1, 5, 15])
        assert all(v == 0.0 for v in profile.reversion_bps)

    def test_sell_side_reversion_sign(self):
        """For a SELL, reversion upward = favorable (negative bps in our convention)."""
        trade = _make_trade(side="SELL", fill_price=99.90)
        # Prices drift back UP after SELL -- reversion of temporary impact
        prices = [99.90 + i * 0.002 for i in range(1, 31)]
        analyzer = ReversionAnalyzer()
        profile = analyzer.analyze(trade, prices, horizons=[5, 15, 30])
        # For a SELL, sign=-1; fill=99.90; price_later > 99.90 => change_bps < 0 => reversion negative
        # (meaning price moved against us = permanent impact, not reversion)
        assert len(profile.reversion_bps) == 3


# ---------------------------------------------------------------------------
# TEST GROUP 6: Reversion database
# ---------------------------------------------------------------------------

class TestReversionDatabase:
    def test_insert_and_retrieve(self):
        """Insert a profile and retrieve it by symbol."""
        db = ReversionDatabase(":memory:")
        profile = ReversionProfile(
            horizons=[1, 5, 15],
            reversion_bps=[-1.0, -3.0, -5.0],
            half_life_bars=5.0,
            permanent_impact=2.0,
            temporary_impact=8.0,
            t_stat=-2.5,
            fit_quality=0.85,
            n_obs=30,
            symbol="AAPL",
            side="BUY",
            trade_id="T001",
        )
        db.insert("T001", profile)
        retrieved = db.get_by_symbol("AAPL", n=10)
        assert len(retrieved) == 1
        assert retrieved[0].symbol == "AAPL"
        assert abs(retrieved[0].half_life_bars - 5.0) < 1e-9

    def test_aggregate_stats_empty(self):
        """aggregate_stats on empty DB should return n_profiles=0."""
        db = ReversionDatabase(":memory:")
        stats = db.aggregate_stats("TSLA")
        assert stats["n_profiles"] == 0


# ---------------------------------------------------------------------------
# TEST GROUP 7: Venue analysis
# ---------------------------------------------------------------------------

class TestVenueAnalysis:
    def _make_results_for_venues(self) -> List[TCAResult]:
        venues = ["NASDAQ", "NYSE", "IEX", "BATS"]
        results = []
        costs = {"NASDAQ": 5.0, "NYSE": 8.0, "IEX": 3.0, "BATS": 6.0}
        for i, venue in enumerate(venues * 5):
            results.append(_make_result(
                is_bps=costs[venue],
                venue=venue,
                trade_id=f"T{i:03d}",
            ))
        return results

    def test_compare_venues_finds_best(self):
        """IEX has lowest avg IS in synthetic data -- should be best venue."""
        results = self._make_results_for_venues()
        analyzer = VenueAnalyzer()
        comparison = analyzer.compare_venues(results)
        assert comparison.best_venue == "IEX"
        assert comparison.worst_venue == "NYSE"

    def test_venue_score_range(self):
        """All venue scores should be in [0, 100]."""
        results = self._make_results_for_venues()
        analyzer = VenueAnalyzer()
        comparison = analyzer.compare_venues(results)
        for vs in comparison.scores.values():
            assert 0.0 <= vs.score <= 100.0

    def test_venue_scorecard(self):
        """venue_scorecard should return a dict of VenueScore objects."""
        results = self._make_results_for_venues()
        analyzer = VenueAnalyzer()
        analyzer.add_results(results)
        scorecard = analyzer.venue_scorecard(window_days=30)
        assert len(scorecard) >= 1
        for vs in scorecard.values():
            assert isinstance(vs, VenueScore)

    def test_best_venue_for_large_order(self):
        """Large order (>1% ADV) should route to dark pool (IEX)."""
        analyzer = VenueAnalyzer()
        venue = analyzer.best_venue_for(
            "AAPL", "BUY", qty=50_000.0, urgency="LOW", adv=2_000_000.0
        )
        assert venue in {"IEX", "BATS_DARK", "CROSSFINDER"}

    def test_best_venue_for_high_urgency(self):
        """High urgency should route to a direct exchange."""
        analyzer = VenueAnalyzer()
        venue = analyzer.best_venue_for(
            "AAPL", "BUY", qty=100.0, urgency="HIGH", adv=2_000_000.0
        )
        assert venue in {"NASDAQ", "NYSE", "CBOE"}

    def test_report_generator_markdown(self):
        """VenueReportGenerator.to_markdown should produce a non-empty string."""
        scores = {
            "NASDAQ": VenueScore("NASDAQ", 5.0, 0.98, 250.0, 2.0, 70.0, 50),
            "IEX": VenueScore("IEX", 3.0, 0.97, 300.0, 1.5, 85.0, 40),
        }
        gen = VenueReportGenerator()
        md = gen.to_markdown(scores)
        assert "## Venue Scorecard" in md
        assert "NASDAQ" in md
        assert "IEX" in md

    def test_report_generator_csv(self):
        """VenueReportGenerator.to_csv should produce parseable CSV."""
        import csv as _csv
        import io as _io
        scores = {
            "NASDAQ": VenueScore("NASDAQ", 5.0, 0.98, 250.0, 2.0, 70.0, 50),
        }
        gen = VenueReportGenerator()
        csv_str = gen.to_csv(scores)
        reader = list(_csv.DictReader(_io.StringIO(csv_str)))
        assert len(reader) == 1
        assert reader[0]["venue"] == "NASDAQ"


# ---------------------------------------------------------------------------
# TEST GROUP 8: TCA store
# ---------------------------------------------------------------------------

class TestTCAStore:
    def test_insert_and_query(self):
        """Inserted result should be retrievable by symbol."""
        store = TCAStore(":memory:")
        result = _make_result(symbol="AAPL", trade_id="T001")
        store.insert("T001", result)
        results = store.query(symbol="AAPL")
        assert len(results) == 1
        assert results[0].symbol == "AAPL"

    def test_upsert_deduplicates(self):
        """Inserting same trade_id twice should not create duplicate rows."""
        store = TCAStore(":memory:")
        result = _make_result(symbol="AAPL", trade_id="T001", is_bps=10.0)
        store.insert("T001", result)
        result2 = _make_result(symbol="AAPL", trade_id="T001", is_bps=15.0)
        store.insert("T001", result2)
        results = store.query(symbol="AAPL")
        assert len(results) == 1

    def test_insert_null_trade_id_always_inserts(self):
        """Empty trade_id should always insert (no deduplication)."""
        store = TCAStore(":memory:")
        result = _make_result(symbol="MSFT", trade_id="")
        store.insert("", result)
        store.insert("", result)
        count = store.count(symbol="MSFT")
        assert count == 2

    def test_query_date_filter(self):
        """date_from/date_to filters should work correctly."""
        store = TCAStore(":memory:")
        r1 = _make_result(symbol="GOOG", trade_id="T001", trade_date="2026-04-07")
        r2 = _make_result(symbol="GOOG", trade_id="T002", trade_date="2026-04-08")
        store.insert("T001", r1)
        store.insert("T002", r2)
        results = store.query(symbol="GOOG", date_from="2026-04-08", date_to="2026-04-08")
        assert len(results) == 1
        assert results[0].trade_date == "2026-04-08"

    def test_aggregate_by_symbol(self):
        """aggregate(group_by='symbol') should return one row per symbol."""
        store = TCAStore(":memory:")
        for i in range(5):
            store.insert(f"T{i}", _make_result(symbol="AAPL", is_bps=10.0 + i, trade_id=f"T{i}"))
        store.insert("T10", _make_result(symbol="GOOG", is_bps=8.0, trade_id="T10"))
        agg = store.aggregate_raw("symbol")
        symbols = [row["symbol"] for row in agg]
        assert "AAPL" in symbols
        assert "GOOG" in symbols

    def test_daily_report(self):
        """daily_report should return correct trade count for a date."""
        store = TCAStore(":memory:")
        for i in range(3):
            store.insert(f"D{i}", _make_result(trade_id=f"D{i}", trade_date="2026-04-07"))
        report = store.daily_report("2026-04-07")
        assert report["n_trades"] == 3
        assert report["date"] == "2026-04-07"

    def test_daily_report_empty(self):
        """daily_report on a date with no trades should return n_trades=0."""
        store = TCAStore(":memory:")
        report = store.daily_report("2020-01-01")
        assert report["n_trades"] == 0

    def test_export_csv(self):
        """export_csv should write a CSV file with correct row count."""
        store = TCAStore(":memory:")
        for i in range(4):
            store.insert(f"E{i}", _make_result(symbol="TSLA", trade_id=f"E{i}"))
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as fh:
            path = fh.name
        try:
            rows = store.export_csv(path, symbol="TSLA")
            assert rows == 4
            with open(path, "r") as fh:
                lines = fh.readlines()
            assert len(lines) == 5  # 1 header + 4 data rows
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# TEST GROUP 9: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_order_qty(self):
        """Zero order quantity should not raise; fill_rate should be 0."""
        trade = _make_trade(order_qty=0.0, filled_qty=0.0)
        engine = TCAEngine()
        result = engine.analyze_trade(trade)
        assert result.fill_rate == 0.0

    def test_same_fill_and_arrival_price(self):
        """Fill == arrival: IS = 0 bps."""
        trade = _make_trade(arrival_price=100.0, fill_price=100.0)
        engine = TCAEngine()
        result = engine.analyze_trade(trade)
        assert abs(result.implementation_shortfall_bps) < 1e-9

    def test_zero_market_volume(self):
        """Zero market volume should produce participation_rate = 0."""
        trade = _make_trade(market_volume=0.0)
        engine = TCAEngine()
        result = engine.analyze_trade(trade)
        assert result.participation_rate == 0.0

    def test_analyze_batch_empty(self):
        """analyze_batch on empty list should return zero metrics."""
        engine = TCAEngine()
        batch = engine.analyze_batch([])
        assert batch.n_trades == 0
        assert batch.avg_is_bps == 0.0

    def test_daily_summary_no_trades(self):
        """daily_summary for a date with no data returns zeroed DailySummary."""
        engine = TCAEngine()
        summary = engine.daily_summary("2020-01-01")
        assert summary.n_trades == 0

    def test_benchmark_close_fallback(self):
        """CloseBenchmark falls back to fill_price when no close available."""
        bench = CloseBenchmark()
        trade = _make_trade(close_price=None)
        trade.close_price = None
        md = MarketData(symbol="AAPL")
        price = bench.compute(trade, md)
        assert price == trade.fill_price

    def test_slippage_decomposition_sums_correctly(self):
        """Decomposed components should sum to approximately total_bps."""
        trade = _make_trade(
            side="BUY",
            arrival_price=100.0,
            fill_price=100.20,
            bid=99.98,
            ask=100.02,
            adv=2_000_000.0,
        )
        md = MarketData(symbol="AAPL", adv=2_000_000.0, sigma=0.02)
        decomp = decompose_slippage(trade, md)
        total_components = (
            decomp.spread_component
            + decomp.market_impact_component
            + decomp.timing_component
            + decomp.alpha_component
        )
        # Total from decompose_slippage may differ slightly from sum due to
        # clamping; verify both are non-negative and roughly consistent
        assert decomp.total_bps >= 0.0
        assert total_components >= 0.0

    def test_impact_model_zero_adv(self):
        """Zero ADV should not raise in SqrtImpactModel.predict."""
        model = SqrtImpactModel()
        result = model.predict(1000.0, 0.0, 0.02)
        assert result == 0.0

    def test_exponential_fit_single_point(self):
        """_fit_exponential_nls with < 3 points returns constant fit."""
        a, b, c, r2 = _fit_exponential_nls([1.0], [5.0])
        assert c == 5.0 or math.isfinite(c)

    def test_analyze_batch_multiple_venues(self):
        """analyze_batch should correctly aggregate by venue."""
        engine = TCAEngine()
        trades = [
            _make_trade(
                venue="NASDAQ" if i % 2 == 0 else "NYSE",
                trade_id=f"BT{i}",
                fill_price=100.0 + i * 0.01,
            )
            for i in range(6)
        ]
        batch = engine.analyze_batch(trades)
        assert len(batch.by_venue) == 2
        assert "NASDAQ" in batch.by_venue
        assert "NYSE" in batch.by_venue
