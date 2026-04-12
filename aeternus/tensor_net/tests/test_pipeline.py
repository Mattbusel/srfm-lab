"""
test_pipeline.py — End-to-end pipeline tests for TensorNet (Project AETERNUS).

Tests the full flow:
  - Data ingestion and preprocessing
  - Normalization and rolling windows
  - Compression pipeline
  - Rank selection
  - Online learning
  - Integration bridges
"""

from __future__ import annotations

import os
import json
import tempfile
import pytest
import numpy as np
import jax.numpy as jnp

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def price_data(rng):
    """Synthetic price data (n_time, n_assets)."""
    n_time, n_assets = 300, 10
    log_prices = np.cumsum(rng.normal(0, 0.01, (n_time, n_assets)), axis=0)
    return np.exp(log_prices).astype(np.float32)


@pytest.fixture
def return_data(price_data):
    from tensor_net.data_pipeline import prices_to_returns
    return prices_to_returns(price_data, return_type="log").astype(np.float32)


@pytest.fixture
def correlation_matrix(return_data):
    """Correlation matrix from return data."""
    corr = np.corrcoef(return_data.T).astype(np.float32)
    return corr


@pytest.fixture
def pipeline_config():
    from tensor_net.data_pipeline import DataPipelineConfig
    return DataPipelineConfig(
        window_size=30,
        stride=5,
        normalize=True,
        normalization_method="zscore",
        augment=False,
        batch_size=16,
        shuffle=False,
        seed=42,
    )


# ============================================================================
# Data pipeline tests
# ============================================================================

class TestDataPipeline:

    def test_prices_to_returns_log(self, price_data):
        from tensor_net.data_pipeline import prices_to_returns
        returns = prices_to_returns(price_data, return_type="log")
        assert returns.shape == (price_data.shape[0] - 1, price_data.shape[1])
        assert np.all(np.isfinite(returns))

    def test_prices_to_returns_simple(self, price_data):
        from tensor_net.data_pipeline import prices_to_returns
        returns = prices_to_returns(price_data, return_type="simple")
        assert returns.shape == (price_data.shape[0] - 1, price_data.shape[1])

    def test_prices_to_returns_clip(self, price_data):
        from tensor_net.data_pipeline import prices_to_returns
        clip = 0.1
        returns = prices_to_returns(price_data, return_type="log", clip=clip)
        assert np.all(np.abs(returns) <= clip + 1e-6)

    def test_fill_missing_zero(self, rng):
        from tensor_net.data_pipeline import fill_missing_returns
        data = rng.normal(0, 0.01, (100, 5)).astype(np.float32)
        data[5, 2] = np.nan
        data[20, 0] = np.nan
        filled = fill_missing_returns(data, method="zero")
        assert np.all(np.isfinite(filled))
        assert filled[5, 2] == 0.0
        assert filled[20, 0] == 0.0

    def test_fill_missing_forward(self, rng):
        from tensor_net.data_pipeline import fill_missing_returns
        data = rng.normal(0, 0.01, (50, 3)).astype(np.float32)
        data[0, :] = 0.5  # Set first row
        data[1, 0] = np.nan
        filled = fill_missing_returns(data, method="forward")
        assert np.isfinite(filled[1, 0])
        assert filled[1, 0] == pytest.approx(0.5, abs=1e-5)

    def test_fill_missing_mean(self, rng):
        from tensor_net.data_pipeline import fill_missing_returns
        data = np.ones((50, 3), dtype=np.float32)
        data[10, 0] = np.nan
        filled = fill_missing_returns(data, method="mean")
        assert np.isfinite(filled[10, 0])

    def test_normalization_zscore(self, return_data):
        from tensor_net.data_pipeline import compute_normalization_stats
        stats = compute_normalization_stats(return_data, method="zscore")
        normalized = stats.normalize(return_data)
        # After normalization, mean should be ~0
        assert np.abs(normalized.mean()) < 0.5

    def test_normalization_minmax(self, return_data):
        from tensor_net.data_pipeline import compute_normalization_stats
        stats = compute_normalization_stats(return_data, method="minmax")
        normalized = stats.normalize(return_data)
        assert normalized.min() >= -0.01
        assert normalized.max() <= 1.01

    def test_normalization_robust(self, return_data):
        from tensor_net.data_pipeline import compute_normalization_stats
        stats = compute_normalization_stats(return_data, method="robust")
        normalized = stats.normalize(return_data)
        assert np.all(np.isfinite(normalized))

    def test_normalization_roundtrip(self, return_data):
        from tensor_net.data_pipeline import compute_normalization_stats
        stats = compute_normalization_stats(return_data, method="zscore")
        norm = stats.normalize(return_data)
        denorm = stats.denormalize(norm)
        assert np.allclose(return_data, denorm, atol=1e-4)

    def test_rolling_window_shape(self, return_data):
        from tensor_net.data_pipeline import build_rolling_window_tensor
        win = 30
        stride = 5
        windows = build_rolling_window_tensor(return_data, win, stride)
        n_expected = (len(return_data) - win) // stride + 1
        assert windows.shape == (n_expected, win, return_data.shape[1])

    def test_rolling_window_no_nan(self, return_data):
        from tensor_net.data_pipeline import build_rolling_window_tensor
        windows = build_rolling_window_tensor(return_data, 30, 5)
        assert np.all(np.isfinite(windows))

    def test_covariance_tensor_shape(self, return_data):
        from tensor_net.data_pipeline import build_covariance_tensor
        win, stride = 20, 5
        cov_tensor = build_covariance_tensor(return_data, win, stride)
        n_assets = return_data.shape[1]
        n_windows = (len(return_data) - win) // stride + 1
        assert cov_tensor.shape == (n_windows, n_assets, n_assets)

    def test_covariance_tensor_symmetric(self, return_data):
        from tensor_net.data_pipeline import build_covariance_tensor
        cov = build_covariance_tensor(return_data, 30, 10)
        for i in range(cov.shape[0]):
            assert np.allclose(cov[i], cov[i].T, atol=1e-5)

    def test_factor_tensor_shapes(self, return_data):
        from tensor_net.data_pipeline import build_factor_tensor
        loadings, factor_rets = build_factor_tensor(return_data, n_factors=3, window_size=30, stride=10)
        n_assets = return_data.shape[1]
        n_windows = (len(return_data) - 30) // 10 + 1
        assert loadings.shape == (n_windows, n_assets, 3)
        assert factor_rets.shape == (n_windows, 30, 3)


# ============================================================================
# Data augmentation tests
# ============================================================================

class TestDataAugmentation:

    def test_noise_injection_shape_preserved(self, rng):
        from tensor_net.data_pipeline import augment_noise_injection
        data = rng.normal(0, 1, (50, 10)).astype(np.float32)
        noisy = augment_noise_injection(data, noise_std=0.01, rng=rng)
        assert noisy.shape == data.shape

    def test_noise_injection_differs(self, rng):
        from tensor_net.data_pipeline import augment_noise_injection
        data = np.ones((50, 10), dtype=np.float32)
        noisy = augment_noise_injection(data, noise_std=0.1, rng=rng)
        assert not np.allclose(data, noisy)

    def test_time_warp_shape(self, rng):
        from tensor_net.data_pipeline import augment_time_warp
        data = rng.normal(0, 1, (30, 5)).astype(np.float32)
        warped = augment_time_warp(data, sigma=0.1, rng=rng)
        assert warped.shape == data.shape

    def test_scaling_shape(self, rng):
        from tensor_net.data_pipeline import augment_scaling
        data = rng.normal(0, 1, (30, 5)).astype(np.float32)
        scaled = augment_scaling(data, scale_range=(0.9, 1.1), rng=rng)
        assert scaled.shape == data.shape

    def test_window_slice_shape(self, rng):
        from tensor_net.data_pipeline import augment_window_slice
        data = rng.normal(0, 1, (30, 5)).astype(np.float32)
        sliced = augment_window_slice(data, min_fraction=0.7, rng=rng)
        assert sliced.shape == data.shape


# ============================================================================
# DataLoader tests
# ============================================================================

class TestDataLoader:

    def test_dataloader_iteration(self, rng):
        from tensor_net.data_pipeline import TensorDataLoader
        data = rng.normal(0, 1, (50, 30, 10)).astype(np.float32)
        loader = TensorDataLoader(data, batch_size=16, shuffle=False)
        batches = list(loader)
        assert len(batches) >= 3

    def test_dataloader_batch_shape(self, rng):
        from tensor_net.data_pipeline import TensorDataLoader
        data = rng.normal(0, 1, (50, 30, 10)).astype(np.float32)
        loader = TensorDataLoader(data, batch_size=16, shuffle=False, drop_last=True)
        for batch in loader:
            assert batch.shape == (16, 30, 10)
            break

    def test_dataloader_shuffle_different_order(self, rng):
        from tensor_net.data_pipeline import TensorDataLoader
        data = rng.normal(0, 1, (100, 20, 5)).astype(np.float32)
        loader = TensorDataLoader(data, batch_size=20, shuffle=True, seed=1)
        batches_1 = [np.array(b) for b in loader]
        loader2 = TensorDataLoader(data, batch_size=20, shuffle=True, seed=2)
        batches_2 = [np.array(b) for b in loader2]
        # Different seeds should produce different orders
        same = all(np.allclose(b1, b2) for b1, b2 in zip(batches_1, batches_2))
        assert not same

    def test_dataloader_n_batches(self, rng):
        from tensor_net.data_pipeline import TensorDataLoader
        data = rng.normal(0, 1, (100, 20, 5)).astype(np.float32)
        loader = TensorDataLoader(data, batch_size=10, shuffle=False)
        assert len(loader) == 10

    def test_dataloader_get_all(self, rng):
        from tensor_net.data_pipeline import TensorDataLoader
        data = rng.normal(0, 1, (30, 20, 5)).astype(np.float32)
        loader = TensorDataLoader(data, batch_size=10)
        all_data = loader.get_all()
        assert all_data.shape == (30, 20, 5)


# ============================================================================
# FinancialDataPipeline end-to-end
# ============================================================================

class TestFinancialDataPipelineEndToEnd:

    def test_pipeline_load_array(self, price_data, pipeline_config):
        from tensor_net.data_pipeline import FinancialDataPipeline
        pipeline = FinancialDataPipeline(pipeline_config)
        pipeline.load_array(price_data)
        pipeline.process()
        assert pipeline.n_windows > 0
        assert pipeline.n_assets == price_data.shape[1]

    def test_pipeline_split_sizes(self, price_data, pipeline_config):
        from tensor_net.data_pipeline import FinancialDataPipeline
        pipeline = FinancialDataPipeline(pipeline_config)
        pipeline.load_array(price_data).process()
        train, val, test = pipeline.split(train_frac=0.7, val_frac=0.15)
        total = len(train) + len(val) + len(test)
        assert total == pipeline.n_windows

    def test_pipeline_dataloader(self, price_data, pipeline_config):
        from tensor_net.data_pipeline import FinancialDataPipeline
        pipeline = FinancialDataPipeline(pipeline_config)
        pipeline.load_array(price_data).process()
        loader = pipeline.get_dataloader("train")
        batches = list(loader)
        assert len(batches) > 0
        for batch in batches:
            assert batch.ndim == 3

    def test_pipeline_denormalize_roundtrip(self, price_data, pipeline_config):
        from tensor_net.data_pipeline import FinancialDataPipeline
        pipeline = FinancialDataPipeline(pipeline_config)
        pipeline.load_array(price_data).process()
        train, _, _ = pipeline.split()
        denorm = pipeline.denormalize(train)
        assert denorm.shape == train.shape

    def test_pipeline_validation_report(self, return_data):
        from tensor_net.data_pipeline import validate_return_tensor
        report = validate_return_tensor(return_data, verbose=False)
        assert "valid" in report
        assert "shape" in report
        assert report["n_nan"] == 0


# ============================================================================
# Compression pipeline tests
# ============================================================================

class TestCompressionPipeline:

    def test_compress_matrix(self, correlation_matrix):
        from tensor_net.compression_pipeline import compress_matrix_to_tt
        compressed = compress_matrix_to_tt(correlation_matrix, max_rank=8)
        assert compressed.compression_ratio > 0
        assert 0 <= compressed.reconstruction_error <= 1.0
        assert len(compressed.cores) >= 1

    def test_decompress_shape(self, correlation_matrix):
        from tensor_net.compression_pipeline import compress_matrix_to_tt, decompress_matrix
        compressed = compress_matrix_to_tt(correlation_matrix, max_rank=8)
        recon = decompress_matrix(compressed)
        assert recon.shape == correlation_matrix.shape

    def test_pipeline_compress_and_decompress(self, correlation_matrix):
        from tensor_net.compression_pipeline import CompressionPipeline
        pipeline = CompressionPipeline()
        compressed = pipeline.compress(correlation_matrix, name="test")
        recon = pipeline.decompress("test")
        assert recon.shape == correlation_matrix.shape

    def test_pipeline_monitor_returns_dict(self, correlation_matrix):
        from tensor_net.compression_pipeline import CompressionPipeline
        pipeline = CompressionPipeline()
        pipeline.compress(correlation_matrix, name="corr")
        monitor = pipeline.monitor()
        assert "stored_matrices" in monitor
        assert "corr" in monitor["stored_matrices"]

    def test_batch_compress(self, rng):
        from tensor_net.compression_pipeline import CompressionPipeline
        pipeline = CompressionPipeline()
        matrices = {
            "A": rng.normal(0, 1, (20, 20)).astype(np.float32),
            "B": rng.normal(0, 1, (30, 30)).astype(np.float32),
        }
        results = pipeline.batch_compress(matrices)
        assert "A" in results
        assert "B" in results

    def test_streaming_compressor(self, rng):
        from tensor_net.compression_pipeline import StreamingCompressor, CompressionPipelineConfig
        config = CompressionPipelineConfig(streaming_window=20, streaming_overlap=5)
        streamer = StreamingCompressor(config)
        for _ in range(25):
            mat = rng.normal(0, 1, (10, 10)).astype(np.float32)
            streamer.update(mat)
        assert streamer.buffer_size <= 20

    def test_compression_quality_metrics(self, correlation_matrix):
        from tensor_net.compression_pipeline import (
            compress_matrix_to_tt, compression_snr, compression_ssim_proxy,
            full_quality_report,
        )
        compressed = compress_matrix_to_tt(correlation_matrix, max_rank=8)
        report = full_quality_report(correlation_matrix, compressed)
        assert "snr_db" in report
        assert "relative_error" in report
        assert "ssim_proxy" in report
        assert 0 <= report["ssim_proxy"] <= 1.0

    def test_pipeline_export_history(self, correlation_matrix, tmp_path):
        from tensor_net.compression_pipeline import CompressionPipeline
        pipeline = CompressionPipeline()
        pipeline.compress(correlation_matrix)
        path = str(tmp_path / "history.json")
        pipeline.export_history_json(path)
        assert os.path.exists(path)
        with open(path) as f:
            records = json.load(f)
        assert len(records) >= 1

    def test_rank1_update(self, correlation_matrix):
        from tensor_net.compression_pipeline import compress_matrix_to_tt, rank1_update_compressed
        compressed = compress_matrix_to_tt(correlation_matrix, max_rank=4)
        n = correlation_matrix.shape[0]
        u = np.ones(n) * 0.01
        v = np.ones(n) * 0.01
        updated = rank1_update_compressed(compressed, u, v, alpha=0.1)
        assert updated is not None


# ============================================================================
# Regime tensor pipeline
# ============================================================================

class TestRegimeTensorPipeline:

    def test_hmm_fit_and_viterbi(self, return_data):
        from tensor_net.regime_tensor import fit_hmm, hmm_viterbi
        obs = return_data[:100]  # first 100 time steps
        params = fit_hmm(obs, n_states=2, n_iter=5)
        seq = hmm_viterbi(obs, params)
        assert len(seq) == len(obs)
        assert set(seq).issubset({0, 1})

    def test_regime_ops_fit(self, return_data):
        from tensor_net.regime_tensor import RegimeTensorOps
        ops = RegimeTensorOps(n_regimes=2, max_rank=4, n_em_iter=3)
        model = ops.fit(return_data[:50], return_data[:50], verbose=False)
        assert model.n_regimes == 2
        assert 0 in model.cores_per_regime
        assert 1 in model.cores_per_regime

    def test_interpolate_regime_cores(self, return_data):
        from tensor_net.regime_tensor import RegimeTensorOps
        ops = RegimeTensorOps(n_regimes=2, max_rank=4, n_em_iter=3)
        ops.fit(return_data[:50], return_data[:50])
        interpolated = ops.interpolate_regime_cores(0, 1, weight_a=0.5)
        assert len(interpolated) > 0

    def test_change_points_list(self, return_data):
        from tensor_net.regime_tensor import RegimeTensorOps
        ops = RegimeTensorOps(n_regimes=2, max_rank=4, n_em_iter=3)
        ops.fit(return_data[:50], return_data[:50])
        cps = ops.detect_change_points(min_segment_length=3)
        assert isinstance(cps, list)

    def test_regime_weighted_prediction(self):
        from tensor_net.regime_tensor import regime_weighted_prediction
        preds = [np.ones(5) * float(r) for r in range(3)]
        weights = np.array([0.5, 0.3, 0.2])
        result = regime_weighted_prediction(weights, preds)
        assert result.shape == (5,)
        expected = 0.5 * 0 + 0.3 * 1 + 0.2 * 2
        assert np.allclose(result, expected, atol=1e-5)


# ============================================================================
# Online learning pipeline
# ============================================================================

class TestOnlineLearningPipeline:

    def test_running_stats_update(self, return_data):
        from tensor_net.online_learning import RunningStats
        n_assets = return_data.shape[1]
        stats = RunningStats(n_assets)
        for row in return_data[:50]:
            stats.update(row)
        assert stats.n == 50
        assert stats.mean.shape == (n_assets,)
        assert np.all(stats.std >= 0)

    def test_running_stats_forget(self, return_data):
        from tensor_net.online_learning import RunningStats
        n = return_data.shape[1]
        stats = RunningStats(n, forget_factor=0.9)
        for row in return_data[:100]:
            stats.update(row)
        # Effective n should be much less than 100 due to forgetting
        assert stats.n < 100

    def test_streaming_svd_update(self, return_data):
        from tensor_net.online_learning import StreamingSVD
        n_features = return_data.shape[1]
        svd = StreamingSVD(rank=4, n_features=n_features, forget_factor=0.99)
        for row in return_data[:50]:
            svd.update(row)
        assert svd.components.shape == (n_features, 4)
        assert svd.singular_values.shape == (4,)

    def test_streaming_svd_reconstruct(self, return_data):
        from tensor_net.online_learning import StreamingSVD
        n_features = return_data.shape[1]
        svd = StreamingSVD(rank=4, n_features=n_features)
        for row in return_data[:50]:
            svd.update(row)
        row = return_data[0]
        recon = svd.reconstruct(row)
        assert recon.shape == row.shape

    def test_rank1_als_update(self, rng):
        from tensor_net.online_learning import rank1_als_update
        cores = [
            rng.normal(0, 0.1, (1, 4, 4)).astype(np.float64),
            rng.normal(0, 0.1, (4, 4, 1)).astype(np.float64),
        ]
        x = rng.normal(0, 1, (16,)).astype(np.float64)
        updated = rank1_als_update(cores, x, learning_rate=0.01)
        assert len(updated) == 2

    def test_memory_bounded_update(self, return_data):
        from tensor_net.online_learning import MemoryBoundedStreamTT
        model = MemoryBoundedStreamTT(max_buffer_size=30, tt_rank=4, recompress_every=10)
        for row in return_data[:50]:
            model.update(row)
        assert model.buffer_size <= 30

    def test_drift_detector_no_drift_stable(self):
        from tensor_net.online_learning import TensorDriftDetector
        detector = TensorDriftDetector(drift_threshold=3.0, min_samples=20)
        drift_events = []
        for _ in range(50):
            result = detector.update(0.1 + np.random.normal(0, 0.01))
            if result["drift"]:
                drift_events.append(1)
        # No drift for stable signal
        assert len(drift_events) == 0

    def test_drift_detector_detects_shift(self):
        from tensor_net.online_learning import TensorDriftDetector
        detector = TensorDriftDetector(drift_threshold=2.0, min_samples=10)
        drift_events = []
        # Stable period
        for _ in range(20):
            detector.update(0.1)
        # Abrupt change
        for _ in range(20):
            result = detector.update(10.0)
            if result["drift"]:
                drift_events.append(1)
        assert len(drift_events) > 0

    def test_online_covariance_tensor(self, return_data):
        from tensor_net.online_learning import OnlineCovarianceTensor
        n_assets = return_data.shape[1]
        oc = OnlineCovarianceTensor(n_assets, rank=4)
        for row in return_data[:50]:
            oc.update(row)
        cov = oc.covariance
        assert cov.shape == (n_assets, n_assets)
        corr = oc.correlation
        assert np.allclose(np.diag(corr), 1.0, atol=0.01)

    def test_drift_aware_online_tt(self, return_data):
        from tensor_net.online_learning import DriftAwareOnlineTT
        model = DriftAwareOnlineTT(
            initial_rank=2, max_rank=16, n_features=return_data.shape[1]
        )
        for row in return_data[:100]:
            result = model.update(row)
        assert model.current_rank >= 2
        summary = model.drift_summary()
        assert "n_seen" in summary


# ============================================================================
# Integration bridge tests
# ============================================================================

class TestIntegrationBridges:

    def test_data_packet_roundtrip_json(self, return_data):
        from tensor_net.integration import make_packet, DataPacket
        packet = make_packet("test", "returns", return_data, metadata={"key": "value"})
        json_str = packet.to_json()
        restored = DataPacket.from_json(json_str)
        assert restored.source == packet.source
        assert restored.data_type == packet.data_type
        assert np.allclose(np.array(restored.payload), return_data, atol=1e-4)

    def test_data_packet_to_dict(self, return_data):
        from tensor_net.integration import make_packet
        packet = make_packet("test", "returns", return_data)
        d = packet.to_dict()
        assert "source" in d
        assert "payload" in d

    def test_validate_packet_valid(self, return_data):
        from tensor_net.integration import make_packet, validate_packet
        packet = make_packet("test", "returns", return_data.astype(np.float32))
        result = validate_packet(packet)
        assert result["valid"]

    def test_validate_packet_nan(self, return_data):
        from tensor_net.integration import make_packet, validate_packet
        bad_data = return_data.copy()
        bad_data[0, 0] = np.nan
        packet = make_packet("test", "returns", bad_data)
        result = validate_packet(packet)
        assert not result["valid"]
        assert any("NaN" in issue for issue in result["issues"])

    def test_lumina_bridge_tokenize(self, return_data):
        from tensor_net.integration import LuminaTokenizerBridge, make_packet
        bridge = LuminaTokenizerBridge(vocab_size=32000, n_bins=256)
        packet = make_packet("test", "returns", return_data[:10])
        tokens = bridge.tensor_to_tokens(packet)
        assert len(tokens) > 0
        assert all(0 <= t < 32000 for t in tokens)

    def test_lumina_bridge_decode(self, rng):
        from tensor_net.integration import LuminaTokenizerBridge, make_packet
        bridge = LuminaTokenizerBridge(vocab_size=32000, n_bins=256)
        data = rng.normal(0, 1, (5, 5)).astype(np.float32)
        packet = make_packet("test", "tensor", data)
        tokens = bridge.tensor_to_tokens(packet)
        decoded = bridge.tokens_to_packet(tokens, shape=(5, 5))
        assert decoded.shape == (5, 5)

    def test_omni_graph_consumer_from_dict(self, tmp_path):
        from tensor_net.integration import OmniGraphConsumer
        graph_data = {
            "nodes": ["A", "B", "C"],
            "edges": [
                {"source": "A", "target": "B", "weight": 0.8},
                {"source": "B", "target": "C", "weight": 0.5},
                {"source": "A", "target": "C", "weight": 0.3},
            ]
        }
        path = str(tmp_path / "graph.json")
        with open(path, "w") as f:
            json.dump(graph_data, f)

        consumer = OmniGraphConsumer(normalize=True, symmetrize=True)
        packet = consumer.consume_json(path)
        assert packet.data_type == "edge_weights"
        assert packet.shape == (3, 3)
        assert np.abs(np.array(packet.payload)).max() <= 1.01

    def test_omni_graph_to_correlation(self, tmp_path):
        from tensor_net.integration import OmniGraphConsumer
        graph_data = {
            "nodes": ["A", "B", "C"],
            "edges": [
                {"source": "A", "target": "B", "weight": 0.8},
                {"source": "B", "target": "C", "weight": 0.5},
            ]
        }
        path = str(tmp_path / "graph.json")
        with open(path, "w") as f:
            json.dump(graph_data, f)

        consumer = OmniGraphConsumer()
        edge_packet = consumer.consume_json(path)
        corr_packet = consumer.edge_weights_to_correlation(edge_packet)
        assert corr_packet.data_type == "correlation"
        corr = np.array(corr_packet.payload)
        assert np.allclose(np.diag(corr), 1.0, atol=1e-3)


# ============================================================================
# Experiment runner tests
# ============================================================================

class TestExperimentRunner:

    def test_comparison_runs(self, rng):
        from tensor_net.experiment_runner import ExperimentRunner
        runner = ExperimentRunner(output_dir="/tmp/test_experiments", verbose=False)
        data = rng.normal(0, 1, (100, 30)).astype(np.float32)
        comparison = runner.compare_methods(data, rank=4, methods=["tt", "tucker", "cp", "full"])
        assert len(comparison.methods) == 4
        for m in ["tt", "tucker", "cp", "full"]:
            assert m in comparison.metrics_per_method

    def test_experiment_runner_save_json(self, rng, tmp_path):
        from tensor_net.experiment_runner import ExperimentRunner
        runner = ExperimentRunner(output_dir=str(tmp_path), verbose=False)
        data = rng.normal(0, 1, (50, 20)).astype(np.float32)

        def my_exp(d, cfg):
            return {"mse": float(d.mean() ** 2), "n_params": cfg.get("rank", 4)}

        runner.run("test_exp", data, {"rank": 4}, my_exp)
        path = runner.save_results_json()
        assert os.path.exists(path)

    def test_config_hash_deterministic(self):
        from tensor_net.experiment_runner import config_hash
        cfg = {"rank": 8, "seed": 42, "method": "bic"}
        h1 = config_hash(cfg)
        h2 = config_hash(cfg)
        assert h1 == h2

    def test_merge_configs(self):
        from tensor_net.experiment_runner import merge_configs
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}, "e": 5}
        merged = merge_configs(base, override)
        assert merged["a"] == 1
        assert merged["b"]["c"] == 99
        assert merged["b"]["d"] == 3
        assert merged["e"] == 5

    def test_set_global_seed(self):
        from tensor_net.experiment_runner import set_global_seed
        key = set_global_seed(42)
        assert key is not None
        # Random samples should be reproducible
        set_global_seed(42)
        x1 = np.random.normal(0, 1, 5)
        set_global_seed(42)
        x2 = np.random.normal(0, 1, 5)
        assert np.allclose(x1, x2)

    def test_result_cache(self, rng, tmp_path):
        from tensor_net.experiment_runner import ResultCache, ExperimentResult
        import time
        cache = ResultCache(str(tmp_path))
        result = ExperimentResult(
            experiment_name="test",
            config={"rank": 4},
            config_hash="abc123",
            timestamp=time.time(),
            elapsed_seconds=1.0,
            metrics={"mse": 0.01},
        )
        cache.store(result)
        loaded = cache.get("abc123", "test")
        assert loaded is not None
        assert loaded["metrics"]["mse"] == pytest.approx(0.01)
