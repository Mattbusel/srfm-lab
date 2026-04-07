"""
test_onchain_pipeline.py -- Test suite for the onchain_pipeline package.

Covers:
  - GlassnodeClient: mock data generation, cache hit/miss, cache TTL
  - OnChainSignalLibrary: signal construction with synthetic data
  - OnChainSignalCombiner: IC weighting, correlation filter, equal weight
  - CryptoRegimeClassifier: regime classification with curated edge-case inputs
  - RegimePositioningAdapter: multiplier lookup and scaling
  - DeFiMonitor: TVL updates, change computation, aggregate flows
  - YieldMonitor: yield queries and carry signal
  - DEXMonitor: volume, liquidity, MEV rate
"""

from __future__ import annotations

import math
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# -- module under test --
import sys
import os
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.onchain_pipeline.glassnode_client import (
    GlassnodeClient,
    GlassnodeCache,
    _generate_mock_series,
)
from research.onchain_pipeline.signal_constructor import (
    OnChainSignalLibrary,
    OnChainSignalCombiner,
)
from research.onchain_pipeline.defi_monitor import (
    DeFiMonitor,
    YieldMonitor,
    DEXMonitor,
)
from research.onchain_pipeline.crypto_regime_classifier import (
    CryptoRegime,
    CryptoRegimeClassifier,
    RegimePositioningAdapter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily_series(
    values: list,
    start: str = "2023-01-01",
    name: str = "test",
) -> pd.Series:
    """Create a UTC-indexed daily pd.Series from a list of values."""
    idx = pd.date_range(start=start, periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=idx, name=name, dtype=float)


def _make_btc_bull_data() -> dict:
    """Synthetic data consistent with a BULL_MARKET regime."""
    n = 300
    prices = _make_daily_series(
        [30_000 + i * 100 for i in range(n)], start="2023-01-01", name="price"
    )
    mvrv_z = _make_daily_series([1.5] * n, name="mvrv_z")
    nupl = _make_daily_series([0.45] * n, name="nupl")
    sopr = _make_daily_series([1.03] * n, name="sopr")
    return dict(prices=prices, mvrv_z=mvrv_z, nupl=nupl, sopr=sopr)


def _make_btc_capitulation_data() -> dict:
    """Synthetic data consistent with CAPITULATION regime."""
    n = 300
    prices = _make_daily_series(
        [50_000 - i * 50 for i in range(n)], start="2023-01-01", name="price"
    )
    mvrv_z = _make_daily_series([-0.8] * n, name="mvrv_z")
    nupl = _make_daily_series([-0.35] * n, name="nupl")
    sopr = _make_daily_series([0.96] * n, name="sopr")
    return dict(prices=prices, mvrv_z=mvrv_z, nupl=nupl, sopr=sopr)


def _make_btc_euphoria_data() -> dict:
    """Synthetic data consistent with EUPHORIA regime."""
    n = 300
    prices = _make_daily_series(
        [50_000 + i * 200 for i in range(n)], start="2023-01-01", name="price"
    )
    mvrv_z = _make_daily_series([4.5] * n, name="mvrv_z")
    nupl = _make_daily_series([0.82] * n, name="nupl")
    sopr = _make_daily_series([1.08] * n, name="sopr")
    funding = _make_daily_series([0.03] * n, name="funding")
    return dict(prices=prices, mvrv_z=mvrv_z, nupl=nupl, sopr=sopr, funding_rate=funding)


# ---------------------------------------------------------------------------
# GlassnodeClient tests
# ---------------------------------------------------------------------------

class TestGlassnodeMockData(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._db_path = Path(self._tmpdir) / "test_cache.db"
        self._client = GlassnodeClient(
            api_key="",
            use_mock=True,
            cache_db_path=self._db_path,
        )

    def tearDown(self):
        self._client.close()

    def test_mvrv_z_returns_series(self):
        series = self._client.mvrv_z_score("BTC")
        self.assertIsInstance(series, pd.Series)
        self.assertGreater(len(series), 0)

    def test_mvrv_z_values_in_realistic_range(self):
        series = self._client.mvrv_z_score("BTC")
        self.assertTrue((series >= -3.0).all(), "MVRV-Z should not be below -3 in mock data")
        self.assertTrue((series <= 8.0).all(), "MVRV-Z should not exceed 8 in mock data")

    def test_sopr_values_in_range(self):
        series = self._client.sopr("BTC")
        self.assertTrue((series >= 0.80).all())
        self.assertTrue((series <= 1.25).all())

    def test_nupl_values_in_range(self):
        series = self._client.nupl("BTC")
        self.assertTrue((series >= -0.6).all())
        self.assertTrue((series <= 1.1).all())

    def test_funding_rate_values_in_range(self):
        series = self._client.funding_rate("BTC")
        self.assertTrue((series.abs() <= 0.06).all())

    def test_series_has_datetime_index(self):
        series = self._client.mvrv_z_score("BTC")
        self.assertIsInstance(series.index, pd.DatetimeIndex)

    def test_all_endpoints_return_data(self):
        methods = [
            self._client.mvrv_z_score,
            self._client.sopr,
            self._client.nupl,
            self._client.exchange_net_flows,
            self._client.long_term_holder_supply,
            self._client.short_term_holder_realized_price,
            self._client.funding_rate,
            self._client.open_interest,
            self._client.realized_cap,
            self._client.thermocap,
        ]
        for method in methods:
            with self.subTest(method=method.__name__):
                result = method("BTC")
                self.assertIsInstance(result, pd.Series)
                self.assertGreater(len(result), 0)

    def test_fetch_all_returns_dataframe(self):
        df = self._client.fetch_all("BTC")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df.columns), 5)


class TestGlassnodeCache(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._db_path = Path(self._tmpdir) / "cache.db"

    def test_cache_miss_returns_none(self):
        cache = GlassnodeCache(db_path=self._db_path, ttl_seconds=3600)
        result, is_stale = cache.get("mvrv_z_score", "BTC", None, None)
        self.assertIsNone(result)
        self.assertFalse(is_stale)
        cache.close()

    def test_cache_set_and_hit(self):
        cache = GlassnodeCache(db_path=self._db_path, ttl_seconds=3600)
        series = _make_daily_series([1.0, 2.0, 3.0], name="mvrv_z_score")
        cache.set("mvrv_z_score", "BTC", None, None, series)

        result, is_stale = cache.get("mvrv_z_score", "BTC", None, None)
        self.assertIsNotNone(result)
        self.assertFalse(is_stale)
        self.assertEqual(len(result), 3)
        cache.close()

    def test_cache_stale_within_revalidate_window(self):
        # TTL of 0 seconds -- immediately stale
        cache = GlassnodeCache(
            db_path=self._db_path,
            ttl_seconds=0,
            stale_revalidate_seconds=3600,
        )
        series = _make_daily_series([1.5, 2.5], name="mvrv_z_score")
        cache.set("mvrv_z_score", "BTC", None, None, series)

        result, is_stale = cache.get("mvrv_z_score", "BTC", None, None)
        self.assertIsNotNone(result)
        self.assertTrue(is_stale)
        cache.close()

    def test_cache_miss_after_stale_window(self):
        # TTL=0, stale window=0 -- anything stored is instantly expired
        cache = GlassnodeCache(
            db_path=self._db_path,
            ttl_seconds=0,
            stale_revalidate_seconds=0,
        )
        series = _make_daily_series([1.0], name="mvrv_z_score")
        cache.set("mvrv_z_score", "BTC", None, None, series)

        # Tiny sleep to ensure timestamp difference
        time.sleep(0.01)
        result, _ = cache.get("mvrv_z_score", "BTC", None, None)
        self.assertIsNone(result)
        cache.close()

    def test_client_uses_cache_on_second_call(self):
        client = GlassnodeClient(
            api_key="",
            use_mock=True,
            cache_db_path=self._db_path,
        )
        s1 = client.mvrv_z_score("BTC")
        s2 = client.mvrv_z_score("BTC")  # should be served from cache
        # Values must match; index freq metadata may differ after round-trip
        np.testing.assert_array_almost_equal(s1.values, s2.values)
        client.close()


# ---------------------------------------------------------------------------
# Signal construction tests
# ---------------------------------------------------------------------------

class TestOnChainSignalLibrary(unittest.TestCase):

    def _lib(self) -> OnChainSignalLibrary:
        return OnChainSignalLibrary()

    def test_mvrv_z_signal_high_value_is_bearish(self):
        """MVRV-Z > 3 should produce a signal near -1."""
        mvrv = _make_daily_series([5.0] * 50, name="mvrv_z")
        lib = self._lib()
        sig = lib.mvrv_z_signal(mvrv)
        self.assertTrue((sig < -0.5).all(), f"High MVRV-Z should be bearish, got {sig.mean():.2f}")

    def test_mvrv_z_signal_low_value_is_bullish(self):
        """MVRV-Z < -1 should produce a signal near +1."""
        mvrv = _make_daily_series([-1.5] * 50, name="mvrv_z")
        lib = self._lib()
        sig = lib.mvrv_z_signal(mvrv)
        self.assertTrue((sig > 0.5).all(), f"Low MVRV-Z should be bullish, got {sig.mean():.2f}")

    def test_nupl_contrarian_euphoria_is_bearish(self):
        nupl = _make_daily_series([0.85] * 50, name="nupl")
        sig = self._lib().nupl_contrarian(nupl)
        self.assertTrue((sig < -0.5).all())

    def test_nupl_contrarian_fear_is_bullish(self):
        nupl = _make_daily_series([-0.2] * 50, name="nupl")
        sig = self._lib().nupl_contrarian(nupl)
        self.assertTrue((sig > 0.5).all())

    def test_sopr_signal_capitulation_is_bullish(self):
        sopr = _make_daily_series([0.93] * 50, name="sopr")
        sig = self._lib().sopr_signal(sopr)
        self.assertTrue((sig > 0.5).all())

    def test_sopr_signal_profit_taking_is_bearish(self):
        sopr = _make_daily_series([1.10] * 50, name="sopr")
        sig = self._lib().sopr_signal(sopr)
        self.assertTrue((sig < -0.5).all())

    def test_funding_rate_signal_high_funding_bearish(self):
        funding = _make_daily_series([0.02] * 50, name="funding")
        sig = self._lib().funding_rate_signal(funding, threshold=0.01)
        self.assertTrue((sig < -0.3).all())

    def test_funding_rate_signal_negative_funding_bullish(self):
        funding = _make_daily_series([-0.02] * 50, name="funding")
        sig = self._lib().funding_rate_signal(funding, threshold=0.01)
        self.assertTrue((sig > 0.3).all())

    def test_all_signals_bounded(self):
        """All signals must stay within [-1, 1]."""
        n = 100
        lib = self._lib()
        rng = np.random.default_rng(42)

        mvrv = _make_daily_series(rng.uniform(-2, 7, n).tolist(), name="mvrv_z")
        nupl = _make_daily_series(rng.uniform(-0.5, 1.0, n).tolist(), name="nupl")
        sopr = _make_daily_series(rng.uniform(0.85, 1.2, n).tolist(), name="sopr")
        flows = _make_daily_series(rng.normal(0, 500, n).tolist(), name="flows")
        lth = _make_daily_series((13_000_000 + np.cumsum(rng.normal(0, 10_000, n))).tolist(), name="lth")
        funding = _make_daily_series(rng.uniform(-0.03, 0.03, n).tolist(), name="funding")

        for name, sig in [
            ("mvrv_z", lib.mvrv_z_signal(mvrv)),
            ("nupl", lib.nupl_contrarian(nupl)),
            ("sopr", lib.sopr_signal(sopr)),
            ("flows", lib.exchange_flow_signal(flows)),
            ("lth", lib.lth_accumulation_signal(lth)),
            ("funding", lib.funding_rate_signal(funding)),
        ]:
            with self.subTest(signal=name):
                valid = sig.dropna()
                self.assertTrue((valid >= -1.0).all(), f"{name} min={valid.min()}")
                self.assertTrue((valid <= 1.0).all(), f"{name} max={valid.max()}")

    def test_oi_signal_with_rising_price_and_oi_bullish(self):
        n = 120  # ensure enough data for rolling zscore warmup
        rng = np.random.default_rng(77)
        oi = _make_daily_series(
            [10e9 + i * 100e6 for i in range(n)], name="oi"
        )
        # Returns must have variance for rolling zscore to work
        returns = _make_daily_series(
            (0.01 + rng.normal(0, 0.002, n)).tolist(), name="returns"
        )
        sig = self._lib().oi_change_signal(oi, returns)
        # After warmup, OI rising + returns positive -> bullish
        valid = sig.dropna()
        self.assertGreater(len(valid), 0)
        # The mean of valid values should be positive given both OI and returns rise
        self.assertGreater(valid.mean(), 0.0)

    def test_empty_series_returns_empty(self):
        lib = self._lib()
        empty = pd.Series(dtype=float, name="mvrv_z")
        result = lib.mvrv_z_signal(empty)
        self.assertTrue(result.empty)


# ---------------------------------------------------------------------------
# Signal combiner tests
# ---------------------------------------------------------------------------

class TestOnChainSignalCombiner(unittest.TestCase):

    def _make_signals(self, n: int = 100) -> dict:
        rng = np.random.default_rng(99)
        return {
            "mvrv_z_signal": _make_daily_series(rng.uniform(-1, 1, n).tolist()),
            "nupl_contrarian": _make_daily_series(rng.uniform(-1, 1, n).tolist()),
            "sopr_signal": _make_daily_series(rng.uniform(-1, 1, n).tolist()),
        }

    def test_equal_weight_combination(self):
        signals = self._make_signals()
        combiner = OnChainSignalCombiner()
        result = combiner.combine(signals, method="equal_weight")
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 100)
        self.assertTrue((result.dropna().abs() <= 1.0).all())

    def test_ic_weight_combination_with_returns(self):
        n = 100
        signals = self._make_signals(n)
        # Create a forward returns series that correlates with one signal
        base = signals["mvrv_z_signal"]
        returns = base * 0.5 + pd.Series(
            np.random.default_rng(7).normal(0, 0.01, n), index=base.index
        )
        combiner = OnChainSignalCombiner(ic_lookback=30)
        result = combiner.combine(signals, method="ic_weight", forward_returns=returns)
        self.assertIsInstance(result, pd.Series)
        self.assertTrue((result.dropna().abs() <= 1.0).all())

    def test_ic_weight_without_returns_uses_equal_weight(self):
        signals = self._make_signals()
        combiner = OnChainSignalCombiner()
        result_ic = combiner.combine(signals, method="ic_weight", forward_returns=None)
        result_eq = combiner.combine(signals, method="equal_weight")
        # Without forward returns, IC defaults to 1.0 for all -- should equal equal_weight
        pd.testing.assert_series_equal(result_ic, result_eq, check_names=False)

    def test_correlation_filter_removes_correlated_signal(self):
        n = 100
        base = pd.Series(np.random.default_rng(55).uniform(-1, 1, n))
        idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
        base.index = idx

        # Nearly identical to base -- should be filtered out
        near_duplicate = base * 0.999 + pd.Series(
            np.random.default_rng(56).normal(0, 0.001, n), index=idx
        )
        signals = {"signal_a": base, "signal_b": near_duplicate}
        combiner = OnChainSignalCombiner(max_corr=0.70)
        _, _, selected = combiner.combine_with_diagnostics(signals, method="equal_weight")
        # One of the two should be dropped due to high correlation
        self.assertEqual(len(selected), 1)

    def test_empty_signals_returns_empty(self):
        combiner = OnChainSignalCombiner()
        result = combiner.combine({})
        self.assertTrue(result.empty)

    def test_rank_weight_method(self):
        signals = self._make_signals()
        combiner = OnChainSignalCombiner()
        result = combiner.combine(signals, method="rank_weight")
        self.assertIsInstance(result, pd.Series)

    def test_invalid_method_raises(self):
        combiner = OnChainSignalCombiner()
        with self.assertRaises(ValueError):
            combiner.combine(self._make_signals(), method="bad_method")


# ---------------------------------------------------------------------------
# Regime classifier tests
# ---------------------------------------------------------------------------

class TestCryptoRegimeClassifier(unittest.TestCase):

    def test_bull_market_classification(self):
        data = _make_btc_bull_data()
        clf = CryptoRegimeClassifier()
        clf.fit(**data)
        regime = clf.classify("2023-10-01")
        self.assertEqual(regime, CryptoRegime.BULL_MARKET)

    def test_capitulation_classification(self):
        data = _make_btc_capitulation_data()
        clf = CryptoRegimeClassifier()
        clf.fit(**data)
        regime = clf.classify("2023-10-01")
        self.assertEqual(regime, CryptoRegime.CAPITULATION)

    def test_euphoria_classification(self):
        data = _make_btc_euphoria_data()
        clf = CryptoRegimeClassifier()
        clf.fit(**data)
        regime = clf.classify("2023-10-01")
        self.assertEqual(regime, CryptoRegime.EUPHORIA)

    def test_unknown_with_no_data(self):
        clf = CryptoRegimeClassifier()
        # No fit called -- should return UNKNOWN
        regime = clf.classify("2023-01-01")
        self.assertEqual(regime, CryptoRegime.UNKNOWN)

    def test_classify_range_returns_series(self):
        data = _make_btc_bull_data()
        clf = CryptoRegimeClassifier()
        clf.fit(**data)
        series = clf.classify_range("2023-06-01", "2023-06-30")
        self.assertIsInstance(series, pd.Series)
        self.assertEqual(len(series), 30)

    def test_regime_durations_sums_to_total_days(self):
        data = _make_btc_bull_data()
        clf = CryptoRegimeClassifier()
        clf.fit(**data)
        durations = clf.regime_durations("2023-06-01", "2023-06-30")
        self.assertEqual(sum(durations.values()), 30)

    def test_bear_market_classification(self):
        n = 300
        # Price far below 200MA: start high then crash
        prices = _make_daily_series(
            [50_000 - i * 300 for i in range(n)], name="price"
        )
        mvrv_z = _make_daily_series([-0.5] * n, name="mvrv_z")
        nupl = _make_daily_series([-0.1] * n, name="nupl")
        sopr = _make_daily_series([0.99] * n, name="sopr")
        clf = CryptoRegimeClassifier()
        clf.fit(prices=prices, mvrv_z=mvrv_z, nupl=nupl, sopr=sopr)
        regime = clf.classify("2023-10-01")
        self.assertIn(regime, [CryptoRegime.BEAR_MARKET, CryptoRegime.CAPITULATION])


# ---------------------------------------------------------------------------
# RegimePositioningAdapter tests
# ---------------------------------------------------------------------------

class TestRegimePositioningAdapter(unittest.TestCase):

    def setUp(self):
        self._adapter = RegimePositioningAdapter()

    def test_bull_market_full_long(self):
        self.assertEqual(self._adapter.long_multiplier(CryptoRegime.BULL_MARKET), 1.0)
        self.assertEqual(self._adapter.short_multiplier(CryptoRegime.BULL_MARKET), 0.0)

    def test_bear_market_only_shorts(self):
        self.assertGreater(self._adapter.short_multiplier(CryptoRegime.BEAR_MARKET), 0.5)
        self.assertLess(self._adapter.long_multiplier(CryptoRegime.BEAR_MARKET), 0.5)

    def test_scale_position_long(self):
        scaled = self._adapter.scale_position(100_000.0, CryptoRegime.BULL_MARKET, is_long=True)
        self.assertEqual(scaled, 100_000.0)

    def test_scale_position_short_in_bear(self):
        scaled = self._adapter.scale_position(100_000.0, CryptoRegime.BEAR_MARKET, is_long=False)
        self.assertGreater(scaled, 50_000.0)

    def test_regime_summary_has_all_regimes(self):
        summary = self._adapter.regime_summary()
        self.assertEqual(len(summary), len(CryptoRegime))

    def test_custom_multipliers(self):
        custom = RegimePositioningAdapter(
            custom_long_multipliers={CryptoRegime.BULL_MARKET: 1.5}
        )
        self.assertEqual(custom.long_multiplier(CryptoRegime.BULL_MARKET), 1.5)


# ---------------------------------------------------------------------------
# DeFiMonitor tests
# ---------------------------------------------------------------------------

class TestDeFiMonitor(unittest.TestCase):

    def setUp(self):
        self._monitor = DeFiMonitor()

    def test_update_and_query_tvl(self):
        self._monitor.update_protocol("aave", {"tvl_usd": 6e9})
        tvl = self._monitor.total_value_locked()
        self.assertIn("aave", tvl)
        self.assertAlmostEqual(tvl["aave"], 6e9)

    def test_aggregate_tvl(self):
        self._monitor.update_protocol("aave", {"tvl_usd": 6e9})
        self._monitor.update_protocol("uniswap", {"tvl_usd": 4e9})
        total = self._monitor.aggregate_tvl()
        self.assertAlmostEqual(total, 10e9)

    def test_tvl_change_positive(self):
        # Backfill historical data then add current
        past = datetime(2023, 1, 1, tzinfo=timezone.utc)
        now_ts = datetime(2023, 1, 15, tzinfo=timezone.utc)
        self._monitor.update_protocol_at("compound", {"tvl_usd": 1e9}, past)
        self._monitor.update_protocol_at("compound", {"tvl_usd": 1.2e9}, now_ts)
        change = self._monitor.tvl_change("compound", window_days=20)
        self.assertAlmostEqual(change, 20.0, places=1)

    def test_unknown_protocol_raises(self):
        with self.assertRaises(KeyError):
            self._monitor.tvl_change("nonexistent_protocol", window_days=7)

    def test_net_tvl_flows_positive(self):
        past = datetime(2023, 1, 1, tzinfo=timezone.utc)
        now_ts = datetime(2023, 1, 15, tzinfo=timezone.utc)
        self._monitor.update_protocol_at("aave", {"tvl_usd": 5e9}, past)
        self._monitor.update_protocol_at("aave", {"tvl_usd": 6e9}, now_ts)
        flows = self._monitor.net_tvl_flows(window_days=20)
        self.assertGreater(flows, 0)

    def test_dominance_sums_correctly(self):
        self._monitor.update_protocol("aave", {"tvl_usd": 6e9})
        self._monitor.update_protocol("uniswap", {"tvl_usd": 4e9})
        dom_aave = self._monitor.dominance("aave")
        dom_uni = self._monitor.dominance("uniswap")
        self.assertAlmostEqual(dom_aave + dom_uni, 1.0)

    def test_extra_fields_stored(self):
        self._monitor.update_protocol("lido", {"tvl_usd": 20e9, "staked_eth": 10_000.0})
        tvl = self._monitor.total_value_locked()
        self.assertIn("lido", tvl)


# ---------------------------------------------------------------------------
# YieldMonitor tests
# ---------------------------------------------------------------------------

class TestYieldMonitor(unittest.TestCase):

    def setUp(self):
        self._monitor = YieldMonitor()

    def test_staking_yield_returns_latest(self):
        self._monitor.update_staking_yield("ETH", 0.035)
        self._monitor.update_staking_yield("ETH", 0.040)
        self.assertAlmostEqual(self._monitor.staking_yield("ETH"), 0.040)

    def test_missing_asset_returns_zero(self):
        self.assertEqual(self._monitor.staking_yield("MISSING"), 0.0)

    def test_lending_rate_query(self):
        self._monitor.update_lending_rate("aave", "USDC", borrow_rate=0.08, lend_rate=0.06)
        self.assertAlmostEqual(self._monitor.lending_rate("aave", "USDC"), 0.06)

    def test_carry_signal_positive_carry(self):
        self._monitor.update_staking_yield("ETH", 0.04)
        self._monitor.update_basis("ETH", 0.01)  # 1% contango
        self._monitor.update_funding_rate("ETH", 0.02)
        carry = self._monitor.carry_signal("ETH")
        # 0.02 + 0.04 - 0.01 = 0.05 positive carry
        self.assertGreater(carry, 0)

    def test_best_lending_yield(self):
        self._monitor.update_lending_rate("aave", "DAI", borrow_rate=0.10, lend_rate=0.07)
        self._monitor.update_lending_rate("compound", "DAI", borrow_rate=0.09, lend_rate=0.065)
        protocol, rate = self._monitor.best_lending_yield("DAI")
        self.assertEqual(protocol, "aave")
        self.assertAlmostEqual(rate, 0.07)


# ---------------------------------------------------------------------------
# DEXMonitor tests
# ---------------------------------------------------------------------------

class TestDEXMonitor(unittest.TestCase):

    def setUp(self):
        self._monitor = DEXMonitor()

    def test_volume_24h_returns_latest(self):
        self._monitor.update_pair("ETH/USDC", volume_24h_usd=500e6, liquidity_usd=2e9)
        self.assertAlmostEqual(self._monitor.volume_24h("ETH/USDC"), 500e6)

    def test_missing_pair_returns_zero(self):
        self.assertEqual(self._monitor.volume_24h("NONEXISTENT/PAIR"), 0.0)

    def test_liquidity_depth_positive(self):
        self._monitor.update_pair("BTC/USDC", volume_24h_usd=100e6, liquidity_usd=500e6)
        depth = self._monitor.liquidity_depth("BTC/USDC", slippage_pct=0.5)
        self.assertGreater(depth, 0)

    def test_volume_to_tvl(self):
        self._monitor.update_protocol("uniswap", volume_24h_usd=1e9, tvl_usd=5e9)
        ratio = self._monitor.volume_to_tvl("uniswap")
        self.assertAlmostEqual(ratio, 0.2)

    def test_sandwich_attack_rate(self):
        self._monitor.update_mev("uniswap", sandwich_count=100, total_tx_count=10_000)
        rate = self._monitor.sandwich_attack_rate("uniswap")
        self.assertAlmostEqual(rate, 0.01)

    def test_zero_tx_count_returns_zero(self):
        self._monitor.update_mev("sushi", sandwich_count=5, total_tx_count=0)
        rate = self._monitor.sandwich_attack_rate("sushi")
        self.assertEqual(rate, 0.0)

    def test_top_pairs_sorted(self):
        self._monitor.update_pair("ETH/USDC", volume_24h_usd=500e6, liquidity_usd=2e9)
        self._monitor.update_pair("BTC/USDC", volume_24h_usd=800e6, liquidity_usd=3e9)
        top = self._monitor.top_pairs_by_volume(n=2)
        self.assertEqual(top[0][0], "BTC/USDC")
        self.assertEqual(top[1][0], "ETH/USDC")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
