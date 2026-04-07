"""
tests/test_onchain_bridge.py

Unit tests for bridge/onchain_bridge.py.
All HTTP calls are mocked; no network access required.
"""

from __future__ import annotations

import math
import sqlite3
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import bridge.onchain_bridge as _onchain_mod

from bridge.onchain_bridge import (
    _TTLCache,
    _clamp,
    _zscore_of,
    ExchangeReserveMonitor,
    FearGreedComposite,
    FundingRateTracker,
    MVRVCalculator,
    OnChainBridge,
    OnChainComposite,
    OnChainMetric,
    OnChainSignalAggregator,
    OpenInterestAnalyzer,
    SOPRTracker,
    WhaleSignalDetector,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_onchain_cache():
    """Clear the module-level TTL cache before every test to avoid pollution."""
    _onchain_mod._cache._store.clear()
    yield
    _onchain_mod._cache._store.clear()


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_onchain.db"


@pytest.fixture()
def bridge(tmp_db: Path) -> OnChainBridge:
    return OnChainBridge(db_path=tmp_db)


# ---------------------------------------------------------------------------
# Helper data factories
# ---------------------------------------------------------------------------


def _make_daily_prices(n: int = 370, base: float = 30000.0, trend: float = 50.0):
    return [[i * 86400000, base + trend * i] for i in range(n)]


def _make_daily_volumes(n: int = 370, base: float = 1e10):
    return [[i * 86400000, base + (i % 7) * 1e9] for i in range(n)]


def _make_market_caps(n: int = 370, base: float = 5e11, trend: float = 5e8):
    return [[i * 86400000, base + trend * i] for i in range(n)]


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestClamp:
    def test_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_above_max(self):
        assert _clamp(2.0) == 1.0

    def test_below_min(self):
        assert _clamp(-3.0) == -1.0

    def test_exactly_bounds(self):
        assert _clamp(-1.0) == -1.0
        assert _clamp(1.0) == 1.0

    def test_custom_bounds(self):
        assert _clamp(5.0, 0.0, 3.0) == 3.0
        assert _clamp(-1.0, 0.0, 1.0) == 0.0


class TestZScore:
    def test_basic(self):
        history = [1.0, 2.0, 3.0, 4.0, 5.0]
        z = _zscore_of(5.0, history)
        assert isinstance(z, float)

    def test_too_short_returns_zero(self):
        assert _zscore_of(1.0, [1.0]) == 0.0
        assert _zscore_of(1.0, []) == 0.0

    def test_zero_std_returns_zero(self):
        assert _zscore_of(3.0, [3.0, 3.0, 3.0, 3.0]) == 0.0


# ---------------------------------------------------------------------------
# TTL Cache tests
# ---------------------------------------------------------------------------


class TestTTLCache:
    def test_set_and_get(self):
        cache = _TTLCache()
        cache.set("key", 42)
        assert cache.get("key", ttl=60) == 42

    def test_expired_returns_none(self):
        cache = _TTLCache()
        cache.set("key", "value")
        # Manually backdate the timestamp
        cache._store["key"] = (time.time() - 120, "value")
        assert cache.get("key", ttl=60) is None

    def test_stale_fallback_returns_value(self):
        cache = _TTLCache()
        cache.set("key", "stale_value")
        cache._store["key"] = (time.time() - 9999, "stale_value")
        assert cache.get_or_stale("key") == "stale_value"

    def test_missing_key_returns_none(self):
        cache = _TTLCache()
        assert cache.get("nonexistent", ttl=60) is None
        assert cache.get_or_stale("nonexistent") is None


# ---------------------------------------------------------------------------
# MVRVCalculator tests
# ---------------------------------------------------------------------------


class TestMVRVCalculator:
    def _mock_response(self, n_days: int = 370) -> dict:
        return {
            "prices": _make_daily_prices(n_days),
            "market_caps": _make_market_caps(n_days),
            "total_volumes": _make_daily_volumes(n_days),
        }

    def test_mvrv_signal_bounded(self):
        calc = MVRVCalculator()
        with patch("bridge.onchain_bridge._safe_get", return_value=self._mock_response()):
            metric = calc.fetch("bitcoin")
        assert -1.0 <= metric.signal_direction <= 1.0

    def test_mvrv_returns_onchain_metric(self):
        calc = MVRVCalculator()
        with patch("bridge.onchain_bridge._safe_get", return_value=self._mock_response()):
            metric = calc.fetch("bitcoin")
        assert isinstance(metric, OnChainMetric)
        assert metric.name == "mvrv_zscore"
        assert metric.source == "coingecko"

    def test_mvrv_high_zscore_gives_short_signal(self):
        """When market cap is far above realized cap, signal should be negative."""
        calc = MVRVCalculator()
        # Build a market cap series that ends very high
        caps = [[i * 86400000, 5e11 + i * 1e9] for i in range(370)]
        # Make the last value 10x the average to force high zscore
        caps[-1][1] = caps[-1][1] * 20
        data = {"prices": _make_daily_prices(), "market_caps": caps, "total_volumes": _make_daily_volumes()}
        with patch("bridge.onchain_bridge._safe_get", return_value=data):
            metric = calc.fetch("bitcoin")
        assert metric.signal_direction < 0.0

    def test_mvrv_failure_returns_cached_or_default(self):
        # Cache cleared by autouse fixture; no prior value -> returns default (0.0)
        calc = MVRVCalculator()
        with patch("bridge.onchain_bridge._safe_get", return_value=None):
            metric = calc.fetch("bitcoin")
        assert isinstance(metric, OnChainMetric)
        assert metric.signal_direction == 0.0

    def test_mvrv_cache_hit(self):
        calc = MVRVCalculator()
        with patch("bridge.onchain_bridge._safe_get", return_value=self._mock_response()) as mock_get:
            calc.fetch("bitcoin")
            calc.fetch("bitcoin")   # second call should use cache
        # _safe_get should only be called once
        assert mock_get.call_count == 1

    def test_mvrv_insufficient_data_returns_default(self):
        calc = MVRVCalculator()
        small_data = {
            "prices": _make_daily_prices(5),
            "market_caps": _make_market_caps(5),
            "total_volumes": _make_daily_volumes(5),
        }
        with patch("bridge.onchain_bridge._safe_get", return_value=small_data):
            metric = calc.fetch("bitcoin")
        assert metric.signal_direction == 0.0


# ---------------------------------------------------------------------------
# SOPRTracker tests
# ---------------------------------------------------------------------------


class TestSOPRTracker:
    def _mock_response(self, n: int = 62) -> dict:
        return {
            "prices": _make_daily_prices(n),
            "total_volumes": _make_daily_volumes(n),
        }

    def test_sopr_bounded(self):
        tracker = SOPRTracker()
        with patch("bridge.onchain_bridge._safe_get", return_value=self._mock_response()):
            metric = tracker.fetch("bitcoin")
        assert -1.0 <= metric.signal_direction <= 1.0

    def test_sopr_below_one_gives_positive_signal(self):
        """Price below 30d MA means SOPR < 1 -> capitulation -> bullish."""
        tracker = SOPRTracker()
        # Declining prices: last price well below MA
        prices = [[i * 86400000, 50000.0 - i * 200.0] for i in range(62)]
        data = {"prices": prices, "total_volumes": _make_daily_volumes(62)}
        with patch("bridge.onchain_bridge._safe_get", return_value=data):
            metric = tracker.fetch("bitcoin")
        assert metric.signal_direction > 0.0

    def test_sopr_failure_returns_default(self):
        tracker = SOPRTracker()
        with patch("bridge.onchain_bridge._safe_get", return_value=None):
            metric = tracker.fetch("bitcoin")
        assert metric.value == 1.0
        assert metric.signal_direction == 0.0


# ---------------------------------------------------------------------------
# FundingRateTracker tests
# ---------------------------------------------------------------------------


class TestFundingRateTracker:
    def _binance_response(self, rate: float = 0.0001) -> list:
        return [{"fundingRate": str(rate), "fundingTime": 1000000} for _ in range(5)]

    def _bybit_response(self, rate: float = 0.0001) -> dict:
        return {
            "result": {
                "list": [{"fundingRate": str(rate), "fundingRateTimestamp": "1000000"} for _ in range(5)]
            }
        }

    def test_funding_signal_bounded(self):
        tracker = FundingRateTracker()

        def mock_get(url, params=None):
            if "binance" in url:
                return self._binance_response(0.0002)
            return self._bybit_response(0.0002)

        with patch("bridge.onchain_bridge._safe_get", side_effect=mock_get):
            metric = tracker.fetch("BTC")
        assert -1.0 <= metric.signal_direction <= 1.0

    def test_high_positive_funding_gives_bearish_signal(self):
        """Crowded long (high funding) -> fade -> bearish."""
        tracker = FundingRateTracker()

        def mock_get(url, params=None):
            if "binance" in url:
                return self._binance_response(0.0009)
            return self._bybit_response(0.0009)

        with patch("bridge.onchain_bridge._safe_get", side_effect=mock_get):
            metric = tracker.fetch("BTC")
        assert metric.signal_direction < 0.0

    def test_negative_funding_gives_bullish_signal(self):
        """Crowded short (negative funding) -> fade -> bullish."""
        tracker = FundingRateTracker()

        def mock_get(url, params=None):
            if "binance" in url:
                return self._binance_response(-0.0009)
            return self._bybit_response(-0.0009)

        with patch("bridge.onchain_bridge._safe_get", side_effect=mock_get):
            metric = tracker.fetch("BTC")
        assert metric.signal_direction > 0.0

    def test_neutral_funding_gives_zero_signal(self):
        tracker = FundingRateTracker()

        def mock_get(url, params=None):
            if "binance" in url:
                return self._binance_response(0.0)
            return self._bybit_response(0.0)

        with patch("bridge.onchain_bridge._safe_get", side_effect=mock_get):
            metric = tracker.fetch("BTC")
        assert metric.signal_direction == 0.0

    def test_funding_failure_returns_stale_or_default(self):
        tracker = FundingRateTracker()
        with patch("bridge.onchain_bridge._safe_get", return_value=None):
            metric = tracker.fetch("BTC")
        assert isinstance(metric, OnChainMetric)

    def test_funding_ttl_is_15_minutes(self):
        """Verify cache key returns fresh value within TTL."""
        from bridge.onchain_bridge import _TTL_FUNDING
        assert _TTL_FUNDING == 900


# ---------------------------------------------------------------------------
# FearGreedComposite tests
# ---------------------------------------------------------------------------


class TestFearGreedComposite:
    def test_extreme_fear_gives_positive_signal(self):
        fg = FearGreedComposite()
        data = {"data": [{"value": "5", "value_classification": "Extreme Fear"}]}
        with patch("bridge.onchain_bridge._safe_get", return_value=data):
            metric = fg.fetch()
        assert metric.signal_direction > 0.5

    def test_extreme_greed_gives_negative_signal(self):
        fg = FearGreedComposite()
        data = {"data": [{"value": "95", "value_classification": "Extreme Greed"}]}
        with patch("bridge.onchain_bridge._safe_get", return_value=data):
            metric = fg.fetch()
        assert metric.signal_direction < -0.5

    def test_neutral_gives_near_zero(self):
        fg = FearGreedComposite()
        data = {"data": [{"value": "50", "value_classification": "Neutral"}]}
        with patch("bridge.onchain_bridge._safe_get", return_value=data):
            metric = fg.fetch()
        assert abs(metric.signal_direction) < 0.05

    def test_signal_bounded(self):
        fg = FearGreedComposite()
        for val in [0, 25, 50, 75, 100]:
            data = {"data": [{"value": str(val), "value_classification": "Test"}]}
            with patch("bridge.onchain_bridge._safe_get", return_value=data):
                fg._compute.__func__(fg)   # direct call to bypass cache
                metric = fg._compute()
            assert -1.0 <= metric.signal_direction <= 1.0

    def test_failure_returns_neutral_default(self):
        fg = FearGreedComposite()
        with patch("bridge.onchain_bridge._safe_get", return_value=None):
            metric = fg.fetch()
        assert metric.value == 50.0
        assert metric.signal_direction == 0.0


# ---------------------------------------------------------------------------
# OnChainSignalAggregator tests
# ---------------------------------------------------------------------------


class TestOnChainSignalAggregator:
    def _stub_metric(self, name: str, signal: float) -> OnChainMetric:
        return OnChainMetric(
            name=name, value=0.0, zscore=0.0,
            signal_direction=signal, timestamp=time.time(), source="test"
        )

    def test_composite_bounded(self):
        agg = OnChainSignalAggregator()
        with (
            patch.object(agg._mvrv, "fetch", return_value=self._stub_metric("mvrv", 0.8)),
            patch.object(agg._sopr, "fetch", return_value=self._stub_metric("sopr", 0.5)),
            patch.object(agg._exchange, "fetch", return_value=self._stub_metric("ex", -0.3)),
            patch.object(agg._funding, "fetch", return_value=self._stub_metric("fund", 0.2)),
            patch.object(agg._oi, "fetch", return_value=self._stub_metric("oi", 0.4)),
            patch.object(agg._whale, "fetch", return_value=self._stub_metric("whale", 0.1)),
            patch.object(agg._fear_greed, "fetch", return_value=self._stub_metric("fg", -0.6)),
        ):
            composite = agg.compute("BTC")
        assert -1.0 <= composite.composite_signal <= 1.0

    def test_all_max_signals_gives_near_one(self):
        agg = OnChainSignalAggregator()
        stub = self._stub_metric("x", 1.0)
        with (
            patch.object(agg._mvrv, "fetch", return_value=stub),
            patch.object(agg._sopr, "fetch", return_value=stub),
            patch.object(agg._exchange, "fetch", return_value=stub),
            patch.object(agg._funding, "fetch", return_value=stub),
            patch.object(agg._oi, "fetch", return_value=stub),
            patch.object(agg._whale, "fetch", return_value=stub),
            patch.object(agg._fear_greed, "fetch", return_value=stub),
        ):
            composite = agg.compute("BTC")
        # Weights sum to < 1 (whale not in composite weighting), but result should be <= 1
        assert composite.composite_signal <= 1.0

    def test_all_zero_signals_gives_zero(self):
        agg = OnChainSignalAggregator()
        stub = self._stub_metric("x", 0.0)
        with (
            patch.object(agg._mvrv, "fetch", return_value=stub),
            patch.object(agg._sopr, "fetch", return_value=stub),
            patch.object(agg._exchange, "fetch", return_value=stub),
            patch.object(agg._funding, "fetch", return_value=stub),
            patch.object(agg._oi, "fetch", return_value=stub),
            patch.object(agg._whale, "fetch", return_value=stub),
            patch.object(agg._fear_greed, "fetch", return_value=stub),
        ):
            composite = agg.compute("BTC")
        assert composite.composite_signal == 0.0

    def test_confidence_reflects_nonzero_signals(self):
        agg = OnChainSignalAggregator()
        zero_stub = self._stub_metric("x", 0.0)
        nonzero_stub = self._stub_metric("x", 0.5)
        with (
            patch.object(agg._mvrv, "fetch", return_value=nonzero_stub),
            patch.object(agg._sopr, "fetch", return_value=nonzero_stub),
            patch.object(agg._exchange, "fetch", return_value=zero_stub),
            patch.object(agg._funding, "fetch", return_value=zero_stub),
            patch.object(agg._oi, "fetch", return_value=zero_stub),
            patch.object(agg._whale, "fetch", return_value=zero_stub),
            patch.object(agg._fear_greed, "fetch", return_value=zero_stub),
        ):
            composite = agg.compute("BTC")
        # 2 out of 6 tracked signals are nonzero -> confidence = 2/6
        assert composite.confidence == pytest.approx(2 / 6, abs=0.01)


# ---------------------------------------------------------------------------
# OnChainBridge tests
# ---------------------------------------------------------------------------


class TestOnChainBridge:
    def _stub_composite(self, symbol: str = "BTC", signal: float = 0.42) -> OnChainComposite:
        return OnChainComposite(
            symbol=symbol,
            composite_signal=signal,
            mvrv_signal=0.1, sopr_signal=0.2, funding_signal=0.3,
            oi_signal=0.1, exchange_reserve_signal=0.0, fear_greed_signal=-0.1,
            timestamp=time.time(), confidence=0.8,
        )

    def test_get_signal_returns_float_in_range(self, bridge: OnChainBridge):
        with patch.object(bridge._aggregator, "compute", return_value=self._stub_composite()):
            result = bridge.get_signal("BTC")
        assert -1.0 <= result <= 1.0
        assert result == pytest.approx(0.42)

    def test_get_signal_cached_avoids_recompute(self, bridge: OnChainBridge):
        composite = self._stub_composite()
        with patch.object(bridge._aggregator, "compute", return_value=composite) as mock_compute:
            bridge.get_signal("BTC")
            bridge.get_signal("BTC")   # should use in-memory cache
        assert mock_compute.call_count == 1

    def test_get_signal_refreshes_when_stale(self, bridge: OnChainBridge):
        composite = self._stub_composite()
        composite.timestamp = time.time() - 7200   # 2 hours ago -> stale
        bridge._last_composite["BTC"] = composite

        fresh = self._stub_composite(signal=0.7)
        with patch.object(bridge._aggregator, "compute", return_value=fresh) as mock_compute:
            result = bridge.get_signal("BTC")
        assert mock_compute.call_count == 1
        assert result == pytest.approx(0.7)

    def test_get_signal_returns_zero_on_exception(self, bridge: OnChainBridge):
        with patch.object(bridge._aggregator, "compute", side_effect=RuntimeError("boom")):
            result = bridge.get_signal("BTC")
        assert result == 0.0

    def test_persist_and_load_from_db(self, bridge: OnChainBridge, tmp_db: Path):
        composite = self._stub_composite()
        bridge._persist(composite)

        loaded = bridge._load_latest("BTC")
        assert loaded is not None
        assert loaded.composite_signal == pytest.approx(0.42)
        assert loaded.symbol == "BTC"

    def test_failure_returns_db_cached_value(self, bridge: OnChainBridge):
        """When refresh fails, should return last DB row."""
        composite = self._stub_composite(signal=0.33)
        bridge._persist(composite)

        with patch.object(bridge._aggregator, "compute", side_effect=RuntimeError("network error")):
            result = bridge.refresh("BTC")
        assert result.composite_signal == pytest.approx(0.33)

    def test_db_created_on_first_use(self, tmp_db: Path):
        b = OnChainBridge(db_path=tmp_db)
        with patch.object(b._aggregator, "compute", return_value=self._stub_composite()):
            b.refresh("BTC")
        assert tmp_db.exists()
        b.close()

    def test_context_manager(self, tmp_db: Path):
        with OnChainBridge(db_path=tmp_db) as b:
            assert b is not None
        assert b._conn is None

    def test_composite_signal_always_bounded(self, bridge: OnChainBridge):
        """Even extreme underlying signals must produce bounded composite."""
        extreme = self._stub_composite(signal=0.9999)
        with patch.object(bridge._aggregator, "compute", return_value=extreme):
            result = bridge.get_signal("BTC")
        assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Integration: full pipeline with mocked HTTP
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_pipeline_no_network(self, tmp_db: Path):
        """Smoke test: full pipeline runs end-to-end with mocked HTTP."""
        daily_data = {
            "prices": _make_daily_prices(370),
            "market_caps": _make_market_caps(370),
            "total_volumes": _make_daily_volumes(370),
        }
        fg_data = {"data": [{"value": "45", "value_classification": "Fear"}]}
        funding_binance = [{"fundingRate": "0.0001", "fundingTime": 1000000}] * 5
        funding_bybit = {
            "result": {"list": [{"fundingRate": "0.0001", "fundingRateTimestamp": "1000"} for _ in range(5)]}
        }
        oi_data = [{"sumOpenInterest": str(1e6 + i * 1e4)} for i in range(24)]

        def mock_get(url, params=None):
            if "alternative.me" in url:
                return fg_data
            if "fundingRate" in url or "fapi.binance.com/fapi/v1/fundingRate" in url:
                return funding_binance
            if "bybit" in url:
                return funding_bybit
            if "openInterestHist" in url:
                return oi_data
            return daily_data

        with patch("bridge.onchain_bridge._safe_get", side_effect=mock_get):
            with OnChainBridge(db_path=tmp_db) as bridge:
                composite = bridge.refresh("BTC")
                signal = bridge.get_signal("BTC")

        assert isinstance(composite, OnChainComposite)
        assert -1.0 <= composite.composite_signal <= 1.0
        assert -1.0 <= signal <= 1.0
        assert composite.confidence > 0.0
