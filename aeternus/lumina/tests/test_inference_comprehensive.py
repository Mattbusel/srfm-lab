"""Comprehensive tests for inference modules."""
import pytest
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lumina"))

# Simple test model
class SimpleModel(nn.Module):
    def __init__(self, d_in=32, d_out=32, vocab_size=100):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_in)
        self.linear = nn.Linear(d_in, d_out)
        self.head = nn.Linear(d_out, vocab_size)
    def forward(self, x):
        h = self.emb(x) if x.dtype == torch.long else self.linear(x)
        return self.head(h)

class TestTokenBuffer:
    def test_put_get(self):
        try:
            from inference import TokenBuffer
            buf = TokenBuffer(maxlen=10)
            for i in range(5):
                buf.put(i)
            assert len(buf.get_all()) == 5
        except ImportError:
            pytest.skip("Not available")

    def test_overflow(self):
        try:
            from inference import TokenBuffer
            buf = TokenBuffer(maxlen=5)
            for i in range(10):
                buf.put(i)
            assert len(buf.get_all()) <= 5
        except ImportError:
            pytest.skip("Not available")

class TestInferenceProfiler:
    @pytest.fixture
    def profiler(self):
        try:
            from inference import InferenceProfiler
            model = SimpleModel()
            return InferenceProfiler(model, device="cpu")
        except ImportError:
            pytest.skip("Not available")

    def test_benchmark_2_16(self, profiler):
        shape = (1, 16, 32)
        result = profiler.benchmark_latency(shape, n_warmup=2, n_benchmark=5)
        assert "mean_ms" in result
        assert result["mean_ms"] > 0

    def test_benchmark_4_32(self, profiler):
        shape = (1, 32, 32)
        result = profiler.benchmark_latency(shape, n_warmup=2, n_benchmark=5)
        assert "mean_ms" in result
        assert result["mean_ms"] > 0

    def test_benchmark_1_64(self, profiler):
        shape = (1, 64, 32)
        result = profiler.benchmark_latency(shape, n_warmup=2, n_benchmark=5)
        assert "mean_ms" in result
        assert result["mean_ms"] > 0

class TestEmbeddingCache:
    @pytest.fixture
    def cache(self):
        try:
            from inference import EmbeddingCache
            return EmbeddingCache(maxsize=10)
        except ImportError:
            pytest.skip("Not available")

    def test_cache_miss_then_hit(self, cache):
        x = torch.randn(2, 8)
        assert cache.get(x) is None
        cache.put(x, "value")
        assert cache.get(x) == "value"

    def test_hit_rate(self, cache):
        x = torch.randn(2, 8)
        cache.get(x)  # miss
        cache.put(x, "v")
        cache.get(x)  # hit
        assert cache.hit_rate == 0.5

    def test_eviction(self, cache):
        for i in range(15):
            x = torch.randn(2, 8)
            cache.put(x, i)
        stats = cache.stats()
        assert stats["cache_size"] <= 10

class TestContextWindowManager:
    @pytest.mark.parametrize("strategy", ["truncate", "sliding", "random_drop"])
    def test_strategies(self, strategy):
        try:
            from inference import ContextWindowManager
            mgr = ContextWindowManager(max_length=100, strategy=strategy)
            tokens = list(range(200))
            result = mgr(tokens)
            assert len(result) <= 100
        except ImportError:
            pytest.skip("Not available")

class TestDynamicQuantizedLinear:
    def test_from_float(self):
        try:
            from inference import DynamicQuantizedLinear
            lin = nn.Linear(64, 32)
            q = DynamicQuantizedLinear.from_float(lin)
            x = torch.randn(2, 16, 64)
            out = q(x)
            assert out.shape == (2, 16, 32)
        except ImportError:
            pytest.skip("Not available")

    def test_output_close_to_float(self):
        try:
            from inference import DynamicQuantizedLinear
            lin = nn.Linear(64, 64)
            q = DynamicQuantizedLinear.from_float(lin)
            x = torch.randn(2, 8, 64)
            with torch.no_grad():
                out_f = lin(x)
                out_q = q(x)
            assert torch.allclose(out_f, out_q, atol=0.1)
        except ImportError:
            pytest.skip("Not available")

class TestGradientBasedAttribution:
    @pytest.fixture
    def attr(self):
        try:
            from inference import GradientBasedAttribution
            model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4))
            return GradientBasedAttribution(model, device="cpu")
        except ImportError:
            pytest.skip("Not available")

    def test_saliency_shape(self, attr):
        x = torch.randn(1, 8, 32)
        sal = attr.saliency(x, target_idx=0)
        assert sal.shape == x.shape

    def test_gradient_x_input_shape(self, attr):
        x = torch.randn(1, 8, 32)
        gxi = attr.gradient_x_input(x, target_idx=0)
        assert gxi.shape == x.shape

    def test_integrated_gradients(self, attr):
        x = torch.randn(1, 8, 32)
        ig = attr.integrated_gradients(x, n_steps=10)
        assert ig.shape == x.shape

    def test_smoothgrad(self, attr):
        x = torch.randn(1, 8, 32)
        sg = attr.smoothgrad(x, n_samples=5)
        assert sg.shape == x.shape

class TestEnsembleInference:
    @pytest.fixture
    def ensemble(self):
        try:
            from inference import EnsembleInference
            models = [nn.Linear(32, 8) for _ in range(3)]
            return EnsembleInference(models, device="cpu")
        except ImportError:
            pytest.skip("Not available")

    def test_predict_mean(self, ensemble):
        x = torch.randn(4, 32)
        out = ensemble.predict_mean(x)
        assert out.shape == (4, 8)

    def test_predict_median(self, ensemble):
        x = torch.randn(4, 32)
        out = ensemble.predict_median(x)
        assert out.shape == (4, 8)

    def test_weight_update(self, ensemble):
        ensemble.update_weights_by_performance([0.9, 0.7, 0.8])
        total = sum(ensemble.weights)
        assert abs(total - 1.0) < 1e-5
