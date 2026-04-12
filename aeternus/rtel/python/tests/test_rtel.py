"""
AETERNUS Real-Time Execution Layer (RTEL)
test_rtel.py — Python tests for shm_reader, feature_store, orchestrator

Run with: python -m pytest python/tests/test_rtel.py -v
"""
from __future__ import annotations

import mmap
import os
import struct
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module imports (adjust sys.path if needed)
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rtel.shm_reader import (
    CACHE_LINE_SIZE, DEFAULT_RING_CAP, DEFAULT_SLOT_BYTES, RTEL_MAGIC,
    RING_CTRL_PADDED, RingControl, SlotHeader, TensorDescriptor,
    ChannelConfig, ChannelCursor, LobSnapshot, ShmChannel, ShmReader,
    DTYPE_FLOAT32, DTYPE_FLOAT64,
)
from rtel.shm_writer import ShmWriter
from rtel.feature_store import FeatureSchema, FeatureStore, FeatureSnapshot
from rtel.experiment_orchestrator import (
    ExperimentConfig, ExperimentOrchestrator,
    MarketSimulator, LuminaStub, HyperAgentStub, PortfolioTracker,
)
from rtel.pipeline_client import PipelineClient, StageMetrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tmpdir_path(tmp_path):
    return tmp_path


@pytest.fixture
def simple_channel(tmpdir_path):
    """Create and return a small ShmChannel for testing."""
    cfg = ChannelConfig(
        name="test.channel",
        slot_bytes=4096,
        ring_capacity=16,
        shm_base_path=tmpdir_path,
        readonly=False,
    )
    # Initialize the file
    writer = ShmWriter(base_path=tmpdir_path)
    writer.open_channel("test.channel", slot_bytes=4096, ring_capacity=16, create=True)
    writer.close()

    ch = ShmChannel(cfg)
    yield ch
    ch.close()


@pytest.fixture
def feature_store():
    return FeatureStore()


@pytest.fixture
def experiment_config():
    return ExperimentConfig(
        name="test_experiment",
        n_assets=3,
        n_steps=50,
        symbols=["AAPL", "GOOGL", "MSFT"],
        seed=42,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# TEST: ShmChannel basic read/write
# ---------------------------------------------------------------------------
class TestShmChannel:

    def test_channel_opens(self, tmpdir_path):
        writer = ShmWriter(base_path=tmpdir_path)
        writer.open_channel("test.open", slot_bytes=4096, ring_capacity=8, create=True)
        writer.close()

        cfg = ChannelConfig(
            name="test.open",
            slot_bytes=4096,
            ring_capacity=8,
            shm_base_path=tmpdir_path,
            readonly=True,
        )
        ch = ShmChannel(cfg)
        assert ch is not None
        ch.close()

    def test_write_and_read(self, tmpdir_path):
        writer = ShmWriter(base_path=tmpdir_path)
        writer.open_channel("test.wr", slot_bytes=4096, ring_capacity=8, create=True)

        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ok = writer.write_array("test.wr", data)
        assert ok, "write should succeed"
        writer.flush_all()

        cfg = ChannelConfig(
            name="test.wr",
            slot_bytes=4096,
            ring_capacity=8,
            shm_base_path=tmpdir_path,
            readonly=True,
        )
        ch = ShmChannel(cfg)
        cur = ChannelCursor(channel_name="test.wr", next_seq=1)

        result = ch.consume(cur)
        assert result is not None, "should have data"
        hdr, arr = result
        assert hdr.is_valid(), "slot should be valid"
        assert len(arr) > 0
        ch.close()
        writer.close()

    def test_multiple_writes_and_reads(self, tmpdir_path):
        writer = ShmWriter(base_path=tmpdir_path)
        writer.open_channel("test.multi", slot_bytes=4096, ring_capacity=32, create=True)

        N = 10
        for i in range(N):
            arr = np.array([float(i)], dtype=np.float32)
            writer.write_array("test.multi", arr)
        writer.flush_all()

        cfg = ChannelConfig(
            name="test.multi",
            slot_bytes=4096,
            ring_capacity=32,
            shm_base_path=tmpdir_path,
            readonly=True,
        )
        ch = ShmChannel(cfg)
        cur = ChannelCursor(channel_name="test.multi", next_seq=1)

        count = 0
        for _ in range(N + 5):
            result = ch.consume(cur)
            if result is None:
                break
            count += 1
        assert count == N, f"expected {N} items, got {count}"
        ch.close()
        writer.close()


# ---------------------------------------------------------------------------
# TEST: ShmWriter
# ---------------------------------------------------------------------------
class TestShmWriter:

    def test_write_predictions(self, tmpdir_path):
        writer = ShmWriter(base_path=tmpdir_path)
        writer.open_channel("aeternus.lumina.predictions",
                            slot_bytes=64*1024, ring_capacity=16, create=True)

        returns    = np.array([0.01, -0.005, 0.003], dtype=np.float32)
        risks      = np.array([0.1, 0.05, 0.07],     dtype=np.float32)
        confidence = np.array([0.8, 0.6, 0.9],        dtype=np.float32)

        ok = writer.write_predictions("aeternus.lumina.predictions",
                                      returns, risks, confidence)
        assert ok
        assert writer.stats("aeternus.lumina.predictions")["published_total"] == 1
        writer.close()

    def test_write_vol_surface(self, tmpdir_path):
        writer = ShmWriter(base_path=tmpdir_path)
        writer.open_channel("vol.test", slot_bytes=64*1024, ring_capacity=8, create=True)
        vols = np.random.uniform(0.1, 0.5, (5, 4))
        ok = writer.write_vol_surface("vol.test", vols, asset_id=0)
        assert ok
        writer.close()

    def test_stats_tracking(self, tmpdir_path):
        writer = ShmWriter(base_path=tmpdir_path)
        writer.open_channel("stats.ch", slot_bytes=4096, ring_capacity=8, create=True)

        for i in range(5):
            writer.write_array("stats.ch", np.array([float(i)], dtype=np.float32))

        stats = writer.stats("stats.ch")
        assert stats["published_total"] == 5
        assert stats["bytes_written"] > 0
        writer.close()


# ---------------------------------------------------------------------------
# TEST: LobSnapshot parsing
# ---------------------------------------------------------------------------
class TestLobSnapshot:

    def _make_flat_array(self) -> np.ndarray:
        """Build a flat f64 array in the Rust publisher format."""
        from rtel.shm_reader import MAX_LOB_LEVELS
        arr = np.zeros(5 + 4 * MAX_LOB_LEVELS + 5, dtype=np.float64)
        arr[0] = 7.0   # asset_id
        arr[1] = 5.0   # n_bids
        arr[2] = 5.0   # n_asks
        arr[3] = time.time_ns()  # timestamp_ns
        arr[4] = 42.0  # sequence

        base_bid_p = 5
        base_bid_s = 5 + MAX_LOB_LEVELS
        base_ask_p = 5 + 2 * MAX_LOB_LEVELS
        base_ask_s = 5 + 3 * MAX_LOB_LEVELS

        for i in range(5):
            arr[base_bid_p + i] = 150.0 - (i+1) * 0.01
            arr[base_bid_s + i] = 100.0
            arr[base_ask_p + i] = 150.0 + (i+1) * 0.01
            arr[base_ask_s + i] = 100.0

        d = 5 + 4 * MAX_LOB_LEVELS
        arr[d]     = 150.0      # mid
        arr[d + 1] = 0.02       # spread
        arr[d + 2] = 0.0        # imbalance (balanced book)
        arr[d + 3] = 150.0 - 0.005  # vwap_bid
        arr[d + 4] = 150.0 + 0.005  # vwap_ask
        return arr

    def test_from_array(self):
        arr = self._make_flat_array()
        snap = LobSnapshot.from_array(arr)
        assert snap.asset_id == 7
        assert snap.sequence == 42
        assert abs(snap.mid_price - 150.0) < 1e-6
        assert abs(snap.spread - 0.02) < 1e-6
        assert abs(snap.bid_imbalance) < 1e-6

    def test_bids_asks_ordering(self):
        arr = self._make_flat_array()
        snap = LobSnapshot.from_array(arr)
        # Best bid should be highest
        if len(snap.bids) > 1:
            assert snap.bids[0][0] >= snap.bids[1][0], "bids should be descending"
        # Best ask should be lowest
        if len(snap.asks) > 1:
            assert snap.asks[0][0] <= snap.asks[1][0], "asks should be ascending"

    def test_5_bid_levels(self):
        arr = self._make_flat_array()
        snap = LobSnapshot.from_array(arr)
        assert len(snap.bids) == 5
        assert len(snap.asks) == 5


# ---------------------------------------------------------------------------
# TEST: FeatureStore
# ---------------------------------------------------------------------------
class TestFeatureStore:

    def test_register_and_get(self, feature_store):
        schema = FeatureSchema.float32("test_feat", (10,), "test feature")
        feature_store.register(schema)
        arr = feature_store.get("test_feat")
        assert arr is not None
        assert arr.shape == (10,)
        assert arr.dtype == np.float32

    def test_update_and_get(self, feature_store):
        schema = FeatureSchema.float64("price", (5,), "prices")
        feature_store.register(schema)
        prices = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)
        ok = feature_store.update("price", prices)
        assert ok
        out = feature_store.get("price")
        np.testing.assert_allclose(out, prices)

    def test_dtype_coercion(self, feature_store):
        schema = FeatureSchema.float32("vals", (3,), "values")
        feature_store.register(schema)
        # Provide float64, should be coerced to float32
        ok = feature_store.update("vals", np.array([1.0, 2.0, 3.0], dtype=np.float64))
        assert ok
        out = feature_store.get("vals")
        assert out.dtype == np.float32

    def test_history_window(self, feature_store):
        schema = FeatureSchema.float32("ts", (1,), "time series")
        feature_store.register(schema)
        for i in range(20):
            feature_store.update("ts", np.array([float(i)], dtype=np.float32))
        hist = feature_store.get_history("ts", 10)
        assert hist is not None
        assert hist.shape[0] == 10

    def test_snapshot(self, feature_store):
        snap = feature_store.snapshot(pipeline_id=42)
        assert snap.version == 1
        assert snap.pipeline_id == 42
        assert isinstance(snap.features, dict)
        assert len(snap.features) > 0

    def test_snapshot_versioning(self, feature_store):
        s1 = feature_store.snapshot()
        s2 = feature_store.snapshot()
        assert s2.version > s1.version

    def test_feature_vector(self, feature_store):
        feature_store.register(FeatureSchema.float32("a", (3,)))
        feature_store.register(FeatureSchema.float32("b", (2,)))
        feature_store.update("a", np.array([1.0, 2.0, 3.0], dtype=np.float32))
        feature_store.update("b", np.array([4.0, 5.0], dtype=np.float32))
        fv = feature_store.feature_vector(["a", "b"])
        assert fv.shape == (5,)
        np.testing.assert_allclose(fv, [1., 2., 3., 4., 5.])

    def test_group_features(self, feature_store):
        lob_names = feature_store.list_features(group="lob")
        assert len(lob_names) > 0
        assert "lob_mid_price" in lob_names

    def test_standard_features_registered(self, feature_store):
        assert feature_store.has("lob_mid_price")
        assert feature_store.has("vol_atm")
        assert feature_store.has("lumina_return_forecast")
        assert feature_store.has("agent_position_delta")

    def test_update_from_lob(self, feature_store):
        snap = LobSnapshot()
        snap.asset_id      = 0
        snap.mid_price     = 123.45
        snap.spread        = 0.05
        snap.bid_imbalance = 0.3
        snap.sequence      = 99
        feature_store.update_from_lob(snap)
        mid = feature_store.get("lob_mid_price")
        assert abs(mid[0] - 123.45) < 1e-10

    def test_save_and_load(self, feature_store, tmp_path):
        feature_store.update(
            "lob_mid_price",
            np.full(512, 42.0, dtype=np.float64))
        path = tmp_path / "features.json"
        feature_store.save(path)
        # Load into fresh store
        store2 = FeatureStore()
        store2.load(path)
        vals = store2.get("lob_mid_price")
        assert vals is not None
        assert abs(vals[0] - 42.0) < 1e-10

    def test_stats(self, feature_store):
        stats = feature_store.stats()
        assert stats["n_features"] > 0
        assert stats["n_groups"] > 0
        assert stats["total_bytes"] > 0


# ---------------------------------------------------------------------------
# TEST: MarketSimulator
# ---------------------------------------------------------------------------
class TestMarketSimulator:

    def test_step_returns_snaps(self, experiment_config):
        sim = MarketSimulator(experiment_config)
        snaps = sim.step()
        assert len(snaps) == experiment_config.n_assets

    def test_prices_positive(self, experiment_config):
        sim = MarketSimulator(experiment_config)
        for _ in range(100):
            sim.step()
        assert all(p > 0 for p in sim.prices)

    def test_lob_levels(self, experiment_config):
        sim = MarketSimulator(experiment_config)
        snaps = sim.step()
        s = snaps[0]
        assert len(s.bids) == experiment_config.n_lob_levels
        assert len(s.asks) == experiment_config.n_lob_levels
        assert s.mid_price > 0
        assert s.spread > 0

    def test_bid_ask_ordering(self, experiment_config):
        sim = MarketSimulator(experiment_config)
        snaps = sim.step()
        s = snaps[0]
        # Best bid < best ask
        assert s.bids[0][0] < s.asks[0][0], "best bid must be < best ask"


# ---------------------------------------------------------------------------
# TEST: LuminaStub
# ---------------------------------------------------------------------------
class TestLuminaStub:

    def test_output_shapes(self):
        stub = LuminaStub(n_assets=5)
        features = np.random.randn(10).astype(np.float32)
        returns, risks, conf = stub.forward(features)
        assert returns.shape == (5,)
        assert risks.shape == (5,)
        assert conf.shape == (5,)

    def test_risks_positive(self):
        stub = LuminaStub(n_assets=5)
        features = np.random.randn(10).astype(np.float32)
        _, risks, _ = stub.forward(features)
        assert np.all(risks > 0), "risks must be positive"

    def test_confidence_bounded(self):
        stub = LuminaStub(n_assets=5)
        for _ in range(100):
            features = np.random.randn(10).astype(np.float32)
            _, _, conf = stub.forward(features)
            assert np.all(conf >= 0) and np.all(conf <= 1), \
                f"confidence out of bounds: {conf}"


# ---------------------------------------------------------------------------
# TEST: PortfolioTracker
# ---------------------------------------------------------------------------
class TestPortfolioTracker:

    def test_basic_pnl(self):
        tracker = PortfolioTracker(n_assets=2, transaction_cost=0.0)
        prices  = np.array([100.0, 200.0])
        returns = np.array([0.01, 0.02])

        # Buy 1 unit of each
        delta = np.array([1.0, 1.0])
        tracker.update(delta, prices, returns)

        assert tracker.cumulative_pnl >= 0  # positive returns, no cost
        assert tracker.n_trades >= 1

    def test_transaction_costs(self):
        tracker = PortfolioTracker(n_assets=2, transaction_cost=0.01)
        prices  = np.array([100.0, 200.0])
        returns = np.array([0.0, 0.0])  # zero returns

        delta = np.array([1.0, 1.0])
        pnl = tracker.update(delta, prices, returns)

        assert pnl < 0, "with zero returns and positive costs, PnL should be negative"
        assert tracker.total_cost > 0

    def test_sharpe_ratio_positive(self):
        tracker = PortfolioTracker(n_assets=1, transaction_cost=0.0)
        prices  = np.array([100.0])
        # Consistent positive returns
        for _ in range(100):
            delta   = np.array([1.0])
            returns = np.array([0.001])
            tracker.update(delta, prices, returns)

        assert tracker.sharpe_ratio > 0

    def test_max_drawdown(self):
        tracker = PortfolioTracker(n_assets=1, transaction_cost=0.0)
        prices = np.array([100.0])
        # Increasing then decreasing returns
        for r in [0.01]*50 + [-0.02]*20:
            tracker.update(np.array([1.0]), prices, np.array([r]))

        assert tracker.max_drawdown > 0


# ---------------------------------------------------------------------------
# TEST: ExperimentOrchestrator end-to-end
# ---------------------------------------------------------------------------
class TestExperimentOrchestrator:

    def test_run_completes(self, experiment_config):
        orch = ExperimentOrchestrator(experiment_config)
        report = orch.run()
        assert report is not None

    def test_report_has_portfolio(self, experiment_config):
        orch = ExperimentOrchestrator(experiment_config)
        report = orch.run()
        assert "sharpe_ratio" in report.portfolio
        assert "cumulative_pnl" in report.portfolio
        assert "n_trades" in report.portfolio

    def test_report_has_latency(self, experiment_config):
        orch = ExperimentOrchestrator(experiment_config)
        report = orch.run()
        assert "pipeline" in report.latency or "lumina" in report.latency

    def test_report_has_accuracy(self, experiment_config):
        orch = ExperimentOrchestrator(experiment_config)
        report = orch.run()
        assert "direction_accuracy" in report.model_accuracy
        acc = report.model_accuracy["direction_accuracy"]
        assert 0.0 <= acc <= 1.0

    def test_n_steps_run(self, experiment_config):
        orch = ExperimentOrchestrator(experiment_config)
        report = orch.run()
        assert report.pipeline_stats["steps_run"] == experiment_config.n_steps

    def test_feature_store_populated(self, experiment_config):
        orch = ExperimentOrchestrator(experiment_config)
        orch.run()
        store = orch.feature_store()
        mid = store.get("lob_mid_price")
        assert mid is not None
        assert mid[0] > 0  # first asset should have a price

    def test_portfolio_tracker_trades(self, experiment_config):
        orch = ExperimentOrchestrator(experiment_config)
        orch.run()
        port = orch.portfolio()
        assert port.n_trades > 0

    def test_report_to_dict(self, experiment_config):
        orch = ExperimentOrchestrator(experiment_config)
        report = orch.run()
        d = report.to_dict()
        assert "config" in d
        assert "portfolio" in d
        assert "latency" in d

    def test_report_save(self, experiment_config, tmp_path):
        experiment_config.output_dir = tmp_path
        orch = ExperimentOrchestrator(experiment_config)
        report = orch.run()
        path = tmp_path / "report.json"
        report.save(path)
        assert path.exists()
        import json
        with open(path) as f:
            d = json.load(f)
        assert "portfolio" in d


# ---------------------------------------------------------------------------
# TEST: PipelineClient (without live shm)
# ---------------------------------------------------------------------------
class TestPipelineClient:

    def test_add_handler(self, tmp_path):
        client = PipelineClient(base_path=tmp_path)
        client.add_handler("test", lambda x: x)
        assert len(client._handlers) == 1
        client.close()

    def test_stage_metrics(self, tmp_path):
        client = PipelineClient(base_path=tmp_path)
        client.add_handler("h1", lambda x: x)
        # Manually record some stage metrics
        m = StageMetrics(name="h1")
        m.record(10_000)
        m.record(20_000)
        assert m.count == 2
        assert m.mean_us == pytest.approx(15.0)
        client.close()

    def test_summary_empty(self, tmp_path):
        client = PipelineClient(base_path=tmp_path)
        s = client.summary()
        assert s["runs"] == 0
        assert s["sla_violations"] == 0
        client.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=False)
    sys.exit(result.returncode)
