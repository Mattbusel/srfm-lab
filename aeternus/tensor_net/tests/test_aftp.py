"""
test_aftp.py — Tests for the Automated Feature-to-Tensor Pipeline (AFTP).
"""

from __future__ import annotations

import time
import pytest
import numpy as np

from tensor_net.aftp_pipeline import (
    # Constants
    DEFAULT_WINDOW,
    COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME, COL_BID, COL_ASK,
    # Data types
    RawTick,
    OHLCVFrame,
    # Rolling state
    RollingState,
    # Feature extractors
    ReturnsExtractor,
    RollingVolatilityExtractor,
    ParkinsonVolatilityExtractor,
    GarmanKlassVolatilityExtractor,
    BidAskFeatureExtractor,
    OrderImbalanceExtractor,
    VWAPDeviationExtractor,
    RSIExtractor,
    MACDExtractor,
    BollingerBandsExtractor,
    # Feature groups
    FeatureGroupConfig,
    build_default_feature_groups,
    # Profiler
    ThroughputProfiler,
    # Pipeline
    AFTPMode,
    AFTPConfig,
    AutomatedFeatureToPipeline,
    create_aftp,
)
from tensor_net.unified_tensor_registry import (
    TensorEnvelope,
    UnifiedTensorRegistry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ASSETS = 4
N_BARS   = 50
WINDOW   = 10


@pytest.fixture
def rng():
    return np.random.default_rng(99)


@pytest.fixture
def tick_array(rng):
    """Single tick: (N_ASSETS, 7) float32."""
    arr = np.zeros((N_ASSETS, 7), dtype=np.float32)
    prices = rng.uniform(50.0, 150.0, N_ASSETS).astype(np.float32)
    arr[:, COL_OPEN]   = prices
    arr[:, COL_HIGH]   = prices * rng.uniform(1.0, 1.01, N_ASSETS)
    arr[:, COL_LOW]    = prices * rng.uniform(0.99, 1.0, N_ASSETS)
    arr[:, COL_CLOSE]  = prices * rng.uniform(0.995, 1.005, N_ASSETS)
    arr[:, COL_VOLUME] = rng.uniform(1e4, 1e6, N_ASSETS)
    arr[:, COL_BID]    = prices - 0.05
    arr[:, COL_ASK]    = prices + 0.05
    return arr


@pytest.fixture
def ohlcv_frame(rng):
    """OHLCVFrame with N_ASSETS assets and N_BARS bars."""
    data = np.zeros((N_ASSETS, N_BARS, 7), dtype=np.float32)
    for i in range(N_ASSETS):
        base = rng.uniform(50.0, 150.0)
        prices = np.cumsum(rng.normal(0.0, 0.5, N_BARS)) + base
        prices = np.clip(prices, 1.0, None).astype(np.float32)
        data[i, :, COL_OPEN]   = prices
        data[i, :, COL_HIGH]   = prices * 1.005
        data[i, :, COL_LOW]    = prices * 0.995
        data[i, :, COL_CLOSE]  = prices * rng.uniform(0.999, 1.001, N_BARS)
        data[i, :, COL_VOLUME] = rng.uniform(1e4, 1e6, N_BARS).astype(np.float32)
        data[i, :, COL_BID]    = prices - 0.05
        data[i, :, COL_ASK]    = prices + 0.05
    timestamps = np.arange(N_BARS, dtype=np.int64)
    asset_ids = [f"ASSET_{i}" for i in range(N_ASSETS)]
    return OHLCVFrame(data=data, timestamps=timestamps, asset_ids=asset_ids, source="test")


@pytest.fixture
def aftp():
    return create_aftp(n_assets=N_ASSETS, t_ticks=32, window=WINDOW)


# ---------------------------------------------------------------------------
# RawTick tests
# ---------------------------------------------------------------------------

class TestRawTick:
    def test_mid(self):
        tick = RawTick(0, 0, 100.0, 101.0, 99.0, 100.5, 1000.0, bid=99.9, ask=100.1)
        assert tick.mid == pytest.approx(100.0, abs=1e-5)

    def test_spread(self):
        tick = RawTick(0, 0, 100.0, 101.0, 99.0, 100.5, 1000.0, bid=99.9, ask=100.1)
        assert tick.spread == pytest.approx(0.2, abs=1e-5)

    def test_imbalance_direction(self):
        tick = RawTick(0, 0, open=100.0, high=105.0, low=99.0, close=104.0,
                       volume=1000.0, bid=99.9, ask=100.1)
        assert tick.imbalance > 0  # bullish


# ---------------------------------------------------------------------------
# RollingState tests
# ---------------------------------------------------------------------------

class TestRollingState:
    def test_push_and_mean(self):
        rs = RollingState(n_assets=2, window=5)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            rs.push(np.array([v, v * 2]))
        m = rs.mean()
        assert m[0] == pytest.approx(3.0, abs=1e-5)
        assert m[1] == pytest.approx(6.0, abs=1e-5)

    def test_std(self):
        rs = RollingState(n_assets=1, window=10)
        data = np.arange(1.0, 11.0)
        for v in data:
            rs.push(np.array([v]))
        assert rs.std()[0] == pytest.approx(np.std(data, ddof=1), abs=1e-4)

    def test_window_overflow(self):
        rs = RollingState(n_assets=1, window=3)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            rs.push(np.array([v]))
        # Window should only hold last 3: [3, 4, 5]
        assert rs.mean()[0] == pytest.approx(4.0, abs=1e-5)

    def test_full_flag(self):
        rs = RollingState(n_assets=1, window=3)
        assert not rs.full()
        for v in [1.0, 2.0]:
            rs.push(np.array([v]))
        assert not rs.full()
        rs.push(np.array([3.0]))
        assert rs.full()

    def test_reset(self):
        rs = RollingState(n_assets=1, window=5)
        rs.push(np.array([1.0]))
        rs.reset()
        assert rs.count == 0
        assert rs.mean()[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Feature extractor tests
# ---------------------------------------------------------------------------

class TestReturnsExtractor:
    def test_first_tick_zero(self, tick_array):
        ext = ReturnsExtractor(N_ASSETS)
        out = ext.update(tick_array)
        assert np.all(out == 0.0)
        assert out.shape == (N_ASSETS, 2)

    def test_second_tick_non_zero(self, tick_array, rng):
        ext = ReturnsExtractor(N_ASSETS)
        ext.update(tick_array)
        tick2 = tick_array.copy()
        tick2[:, COL_CLOSE] *= 1.01
        out = ext.update(tick2)
        assert not np.all(out == 0.0)

    def test_reset_clears_state(self, tick_array, rng):
        ext = ReturnsExtractor(N_ASSETS)
        ext.update(tick_array)
        ext.reset()
        out = ext.update(tick_array)
        assert np.all(out == 0.0)

    def test_log_return_positive_for_up_move(self, tick_array, rng):
        ext = ReturnsExtractor(N_ASSETS)
        ext.update(tick_array)
        tick2 = tick_array.copy()
        tick2[:, COL_CLOSE] = tick_array[:, COL_CLOSE] * 1.05
        out = ext.update(tick2)
        assert np.all(out[:, 1] > 0.0)  # log returns should be positive


class TestRollingVolatilityExtractor:
    def test_output_shape(self, tick_array):
        ext = RollingVolatilityExtractor(N_ASSETS, window=WINDOW)
        out = ext.update(tick_array)
        assert out.shape == (N_ASSETS, 1)

    def test_volatility_positive(self, rng):
        ext = RollingVolatilityExtractor(N_ASSETS, window=5)
        prev = None
        for _ in range(10):
            tick = np.zeros((N_ASSETS, 7), dtype=np.float32)
            tick[:, COL_CLOSE] = rng.uniform(90.0, 110.0, N_ASSETS).astype(np.float32)
            out = ext.update(tick)
            prev = out
        assert np.all(prev >= 0.0)


class TestParkinsonVolatilityExtractor:
    def test_output_dtype(self, tick_array):
        ext = ParkinsonVolatilityExtractor(N_ASSETS, window=WINDOW)
        out = ext.update(tick_array)
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))

    def test_higher_hl_means_higher_vol(self, rng):
        ext_low  = ParkinsonVolatilityExtractor(1, window=5)
        ext_high = ParkinsonVolatilityExtractor(1, window=5)
        for _ in range(5):
            base = rng.uniform(100.0, 101.0)
            tick_low = np.zeros((1, 7), dtype=np.float32)
            tick_low[:, COL_HIGH] = base + 0.1
            tick_low[:, COL_LOW]  = base - 0.1
            tick_low[:, COL_CLOSE] = base
            tick_low[:, COL_OPEN]  = base

            tick_high = np.zeros((1, 7), dtype=np.float32)
            tick_high[:, COL_HIGH] = base + 5.0
            tick_high[:, COL_LOW]  = base - 5.0
            tick_high[:, COL_CLOSE] = base
            tick_high[:, COL_OPEN]  = base

            out_low  = ext_low.update(tick_low)
            out_high = ext_high.update(tick_high)
        assert float(out_high[0, 0]) > float(out_low[0, 0])


class TestGarmanKlassExtractor:
    def test_output_shape(self, tick_array):
        ext = GarmanKlassVolatilityExtractor(N_ASSETS, window=WINDOW)
        out = ext.update(tick_array)
        assert out.shape == (N_ASSETS, 1)
        assert np.all(out >= 0.0)


class TestBidAskFeatureExtractor:
    def test_output_shape(self, tick_array):
        ext = BidAskFeatureExtractor(N_ASSETS, window=WINDOW)
        out = ext.update(tick_array)
        assert out.shape == (N_ASSETS, 4)

    def test_spread_is_positive(self, tick_array):
        ext = BidAskFeatureExtractor(N_ASSETS)
        out = ext.update(tick_array)
        assert np.all(out[:, 0] >= 0.0)

    def test_mid_between_bid_ask(self, tick_array):
        ext = BidAskFeatureExtractor(N_ASSETS)
        out = ext.update(tick_array)
        mid = out[:, 1]
        bid = tick_array[:, COL_BID]
        ask = tick_array[:, COL_ASK]
        expected_mid = (bid + ask) / 2.0
        assert np.allclose(mid, expected_mid, atol=1e-4)


class TestOrderImbalanceExtractor:
    def test_output_shape(self, tick_array):
        ext = OrderImbalanceExtractor(N_ASSETS)
        out = ext.update(tick_array)
        assert out.shape == (N_ASSETS, 2)

    def test_imbalance_range(self, tick_array):
        ext = OrderImbalanceExtractor(N_ASSETS)
        out = ext.update(tick_array)
        assert np.all(out[:, 0] >= -1.0 - 1e-6)
        assert np.all(out[:, 0] <= 1.0 + 1e-6)


class TestVWAPDeviationExtractor:
    def test_output_shape(self, tick_array):
        ext = VWAPDeviationExtractor(N_ASSETS, window=5)
        out = ext.update(tick_array)
        assert out.shape == (N_ASSETS, 2)

    def test_vwap_positive(self, tick_array):
        ext = VWAPDeviationExtractor(N_ASSETS)
        out = ext.update(tick_array)
        assert np.all(out[:, 0] > 0.0)


class TestRSIExtractor:
    def test_initial_rsi_50(self, tick_array):
        ext = RSIExtractor(N_ASSETS, period=14)
        out = ext.update(tick_array)
        assert np.allclose(out, 50.0, atol=1.0)

    def test_rsi_range_after_many_ticks(self, rng):
        ext = RSIExtractor(1, period=7)
        for _ in range(50):
            tick = np.zeros((1, 7), dtype=np.float32)
            tick[:, COL_CLOSE] = rng.uniform(80.0, 120.0, 1).astype(np.float32)
            out = ext.update(tick)
        assert 0.0 <= float(out[0, 0]) <= 100.0

    def test_rsi_high_on_up_trend(self):
        ext = RSIExtractor(1, period=7)
        for i in range(30):
            tick = np.zeros((1, 7), dtype=np.float32)
            tick[:, COL_CLOSE] = np.array([[100.0 + i]])
            out = ext.update(tick)
        # Strong uptrend -> RSI should be high
        assert float(out[0, 0]) > 70.0


class TestMACDExtractor:
    def test_output_shape(self, tick_array):
        ext = MACDExtractor(N_ASSETS)
        out = ext.update(tick_array)
        assert out.shape == (N_ASSETS, 3)

    def test_initial_zeros(self, tick_array):
        ext = MACDExtractor(N_ASSETS)
        out = ext.update(tick_array)
        assert np.all(out == 0.0)

    def test_values_after_ticks(self, rng):
        ext = MACDExtractor(1)
        for _ in range(30):
            tick = np.zeros((1, 7), dtype=np.float32)
            tick[:, COL_CLOSE] = rng.uniform(90.0, 110.0, 1).astype(np.float32)
            out = ext.update(tick)
        assert np.all(np.isfinite(out))


class TestBollingerBandsExtractor:
    def test_output_shape(self, tick_array):
        ext = BollingerBandsExtractor(N_ASSETS, window=WINDOW)
        out = ext.update(tick_array)
        assert out.shape == (N_ASSETS, 5)

    def test_band_ordering(self, rng):
        ext = BollingerBandsExtractor(1, window=5, n_std=2.0)
        for _ in range(5):
            tick = np.zeros((1, 7), dtype=np.float32)
            tick[:, COL_CLOSE] = rng.uniform(90.0, 110.0).astype(np.float32)
            out = ext.update(tick)
        # upper >= mid >= lower
        assert float(out[0, 0]) >= float(out[0, 1])
        assert float(out[0, 1]) >= float(out[0, 2])


# ---------------------------------------------------------------------------
# Feature group config
# ---------------------------------------------------------------------------

class TestFeatureGroupConfig:
    def test_build_default_groups(self):
        groups = build_default_feature_groups(N_ASSETS, window=WINDOW)
        assert len(groups) > 0
        for g in groups:
            assert g.total_output_dim > 0
            assert g.enabled

    def test_group_names_unique(self):
        groups = build_default_feature_groups(N_ASSETS)
        names = [g.name for g in groups]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# ThroughputProfiler
# ---------------------------------------------------------------------------

class TestThroughputProfiler:
    def test_basic_report(self):
        p = ThroughputProfiler()
        p.start()
        for i in range(5):
            p.record_group("returns", 1_000)
            p.tick_done()
        report = p.report()
        assert report.total_ticks == 5
        assert report.ticks_per_sec > 0.0
        assert "returns" in report.group_latency_us
        assert report.group_latency_us["returns"] == pytest.approx(1.0, abs=0.1)

    def test_reset(self):
        p = ThroughputProfiler()
        p.start()
        p.tick_done()
        p.reset()
        report = p.report()
        assert report.total_ticks == 0


# ---------------------------------------------------------------------------
# AFTP streaming mode
# ---------------------------------------------------------------------------

class TestAFTPStreaming:
    def test_push_tick_returns_envelope(self, aftp, tick_array):
        aftp.start()
        env = aftp.push_tick(tick_array)
        aftp.stop()
        assert isinstance(env, TensorEnvelope)
        assert env.schema_name == "ChronosOutput"

    def test_push_tick_output_shape(self, aftp, tick_array):
        aftp.start()
        env = aftp.push_tick(tick_array)
        aftp.stop()
        assert env.data.shape[0] == N_ASSETS
        assert env.data.shape[2] == 6

    def test_push_tick_dtype(self, aftp, tick_array):
        aftp.start()
        env = aftp.push_tick(tick_array)
        aftp.stop()
        assert env.data.dtype == np.float32

    def test_tick_id_increments(self, aftp, tick_array):
        aftp.start()
        for i in range(5):
            aftp.push_tick(tick_array)
        aftp.stop()
        assert aftp.tick_id == 5

    def test_buffer_filled_after_window(self, rng):
        t_ticks = 8
        aftp = create_aftp(n_assets=N_ASSETS, t_ticks=t_ticks, window=4)
        aftp.start()
        assert not aftp.buffer_filled
        for _ in range(t_ticks):
            tick = np.zeros((N_ASSETS, 7), dtype=np.float32)
            tick[:, COL_CLOSE] = rng.uniform(90.0, 110.0, N_ASSETS).astype(np.float32)
            tick[:, COL_BID]   = tick[:, COL_CLOSE] - 0.05
            tick[:, COL_ASK]   = tick[:, COL_CLOSE] + 0.05
            tick[:, COL_HIGH]  = tick[:, COL_CLOSE] + 0.1
            tick[:, COL_LOW]   = tick[:, COL_CLOSE] - 0.1
            tick[:, COL_OPEN]  = tick[:, COL_CLOSE]
            tick[:, COL_VOLUME] = 1e5
            aftp.push_tick(tick)
        aftp.stop()
        assert aftp.buffer_filled

    def test_push_wrong_shape_raises(self, aftp):
        aftp.start()
        bad_tick = np.zeros((N_ASSETS, 5), dtype=np.float32)
        with pytest.raises(ValueError):
            aftp.push_tick(bad_tick)
        aftp.stop()

    def test_reset_state(self, aftp, tick_array):
        aftp.start()
        for _ in range(3):
            aftp.push_tick(tick_array)
        aftp.reset_state()
        assert aftp.tick_id == 0
        assert not aftp.buffer_filled
        aftp.stop()

    def test_profiler_report(self, aftp, tick_array):
        aftp.start()
        for _ in range(10):
            aftp.push_tick(tick_array)
        report = aftp.profiler_report()
        aftp.stop()
        assert report is not None
        assert report.total_ticks == 10
        assert report.ticks_per_sec > 0.0

    def test_no_nan_in_output(self, aftp, tick_array):
        aftp.start()
        env = aftp.push_tick(tick_array)
        aftp.stop()
        assert not np.any(np.isnan(env.data))

    def test_bid_leq_ask_in_output(self, aftp, tick_array):
        aftp.start()
        env = aftp.push_tick(tick_array)
        aftp.stop()
        bid = env.data[:, -1, 0]
        ask = env.data[:, -1, 1]
        assert np.all(bid <= ask + 1e-5)


# ---------------------------------------------------------------------------
# AFTP batch mode
# ---------------------------------------------------------------------------

class TestAFTPBatch:
    def test_process_batch_shape(self, aftp, ohlcv_frame):
        env = aftp.process_batch(ohlcv_frame)
        assert env.data.shape == (N_ASSETS, N_BARS, 6)

    def test_process_batch_dtype(self, aftp, ohlcv_frame):
        env = aftp.process_batch(ohlcv_frame)
        assert env.data.dtype == np.float32

    def test_process_batch_no_nan(self, aftp, ohlcv_frame):
        env = aftp.process_batch(ohlcv_frame)
        assert not np.any(np.isnan(env.data))

    def test_process_batch_wrong_assets_raises(self, rng):
        aftp = create_aftp(n_assets=N_ASSETS, t_ticks=32)
        bad_data = np.zeros((N_ASSETS + 1, N_BARS, 7), dtype=np.float32)
        bad_frame = OHLCVFrame(
            data=bad_data,
            timestamps=np.arange(N_BARS),
            asset_ids=[f"A{i}" for i in range(N_ASSETS + 1)],
        )
        with pytest.raises(ValueError):
            aftp.process_batch(bad_frame)

    def test_process_batch_wrong_columns_raises(self, rng):
        aftp = create_aftp(n_assets=N_ASSETS, t_ticks=32)
        bad_data = np.zeros((N_ASSETS, N_BARS, 5), dtype=np.float32)
        bad_frame = OHLCVFrame(
            data=bad_data,
            timestamps=np.arange(N_BARS),
            asset_ids=[f"A{i}" for i in range(N_ASSETS)],
        )
        with pytest.raises(ValueError):
            aftp.process_batch(bad_frame)

    def test_process_batch_schema_name(self, aftp, ohlcv_frame):
        env = aftp.process_batch(ohlcv_frame)
        assert env.schema_name == "ChronosOutput"

    def test_process_batch_metadata(self, aftp, ohlcv_frame):
        env = aftp.process_batch(ohlcv_frame)
        assert env.metadata.get("n_bars") == N_BARS

    def test_process_batch_tick_id_advances(self, aftp, ohlcv_frame):
        aftp.process_batch(ohlcv_frame)
        aftp.process_batch(ohlcv_frame)
        assert aftp.tick_id == N_BARS * 2


# ---------------------------------------------------------------------------
# AFTP create_aftp factory
# ---------------------------------------------------------------------------

class TestCreateAFTP:
    def test_default_config(self):
        aftp = create_aftp(n_assets=5, t_ticks=16)
        assert isinstance(aftp, AutomatedFeatureToPipeline)

    def test_streaming_mode(self):
        aftp = create_aftp(n_assets=5, t_ticks=16, mode=AFTPMode.STREAMING)
        assert aftp._cfg.mode == AFTPMode.STREAMING

    def test_profiler_disabled(self):
        aftp = create_aftp(n_assets=5, t_ticks=16, enable_profiler=False)
        assert aftp.profiler_report() is None

    def test_validate_output_disabled(self, tick_array):
        aftp = create_aftp(n_assets=N_ASSETS, t_ticks=32, validate_output=False)
        aftp.start()
        env = aftp.push_tick(tick_array)
        aftp.stop()
        assert env is not None
