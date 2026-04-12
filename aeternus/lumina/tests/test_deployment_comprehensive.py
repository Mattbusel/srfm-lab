"""Tests for deployment.py components."""
import pytest
import time
import torch
import torch.nn as nn

class TestServingConfig:
    def test_defaults(self):
        from deployment import ServingConfig
        cfg = ServingConfig()
        assert cfg.max_batch_size == 32
        assert cfg.device == 'cpu'

    def test_custom(self):
        from deployment import ServingConfig
        cfg = ServingConfig(max_batch_size=8, timeout_ms=50.0, use_fp16=False)
        assert cfg.max_batch_size == 8

class TestRequest:
    def test_creation(self):
        from deployment import Request
        req = Request('req1', {'x': torch.randn(4)})
        assert req.request_id == 'req1'
        assert req.priority == 0
        assert req.timestamp > 0

class TestRequestQueue:
    def test_put_and_get(self):
        from deployment import RequestQueue, Request
        q = RequestQueue()
        req = Request('r1', {'x': torch.randn(4)}, priority=1)
        q.put(req)
        out = q.get(block=False)
        assert out.request_id == 'r1'

    def test_priority_ordering(self):
        from deployment import RequestQueue, Request
        q = RequestQueue()
        q.put(Request('low', {}, priority=0))
        q.put(Request('high', {}, priority=10))
        first = q.get(block=False)
        assert first.request_id == 'high'

    def test_get_batch(self):
        from deployment import RequestQueue, Request
        q = RequestQueue()
        for i in range(5):
            q.put(Request(f'r{i}', {}))
        batch = q.get_batch(max_size=3, timeout_ms=10.0)
        assert len(batch) == 3

class TestDynamicBatcher:
    def test_batch_on_size(self):
        from deployment import DynamicBatcher, Request
        batcher = DynamicBatcher(max_batch_size=3, timeout_ms=1000.0)
        result = None
        for i in range(3):
            result = batcher.add(Request(f'r{i}', {}))
        assert result is not None
        assert len(result) == 3

    def test_flush(self):
        from deployment import DynamicBatcher, Request
        batcher = DynamicBatcher(max_batch_size=10)
        batcher.add(Request('r1', {}))
        batcher.add(Request('r2', {}))
        batch = batcher.flush()
        assert len(batch) == 2

class TestInferenceCache:
    def test_miss_then_hit(self):
        from deployment import InferenceCache, Response
        cache = InferenceCache(capacity=100)
        inputs = {'x': torch.tensor([1.0, 2.0, 3.0])}
        assert cache.get(inputs) is None
        resp = Response('r1', {'out': torch.tensor([1.0])})
        cache.put(inputs, resp)
        hit = cache.get(inputs)
        assert hit is not None
        assert hit.request_id == 'r1'

    def test_capacity_limit(self):
        from deployment import InferenceCache, Response
        cache = InferenceCache(capacity=5)
        for i in range(10):
            inp = {'x': torch.tensor([float(i)])};
            cache.put(inp, Response(f'r{i}', {}))
        assert len(cache._cache) <= 5

    def test_hit_rate(self):
        from deployment import InferenceCache, Response
        cache = InferenceCache(capacity=10)
        inp = {'x': torch.tensor([1.0, 2.0])}
        cache.get(inp)  # miss
        cache.put(inp, Response('r1', {}))
        cache.get(inp)  # hit
        assert cache.hit_rate > 0

class TestServerMetrics:
    def test_record_and_summary(self):
        from deployment import ServerMetrics
        m = ServerMetrics()
        for i in range(10):
            m.record_batch(4, float(10 + i))
        s = m.summary()
        assert 'p50_latency_ms' in s
        assert s['total_requests'] == 40

    def test_empty_summary(self):
        from deployment import ServerMetrics
        m = ServerMetrics()
        s = m.summary()
        assert 'total_requests' in s

class TestTokenBucketRateLimiter:
    def test_acquire_within_burst(self):
        from deployment import TokenBucketRateLimiter
        limiter = TokenBucketRateLimiter(rate_qps=100.0, burst=10.0)
        result = limiter.acquire(block=False)
        assert result is True

    def test_acquire_exceed_burst_nonblocking(self):
        from deployment import TokenBucketRateLimiter
        limiter = TokenBucketRateLimiter(rate_qps=1.0, burst=1.0)
        limiter.acquire(1.0, block=False)  # use up burst
        result = limiter.acquire(1.0, block=False)
        assert result is False

class TestModelVersionManager:
    def test_register_and_route(self):
        from deployment import ModelVersionManager
        vm = ModelVersionManager()
        m1 = nn.Linear(4, 2)
        m2 = nn.Linear(4, 2)
        vm.register('v1', m1, traffic_weight=0.8)
        vm.register('v2', m2, traffic_weight=0.2)
        version = vm.route_request()
        assert version in ['v1', 'v2']

    def test_canary_deployment(self):
        from deployment import ModelVersionManager
        vm = ModelVersionManager()
        m1 = nn.Linear(4, 2)
        m2 = nn.Linear(4, 2)
        vm.register('v1', m1)
        vm.register('v2', m2)
        vm.set_canary('v2', canary_fraction=0.1)
        assert abs(vm._traffic_weights['v2'] - 0.1) < 0.01

    def test_promote_canary(self):
        from deployment import ModelVersionManager
        vm = ModelVersionManager()
        vm.register('v1', nn.Linear(4, 2))
        vm.register('v2', nn.Linear(4, 2))
        vm.promote_canary('v2')
        assert vm._active_version == 'v2'
        assert vm._traffic_weights['v2'] == 1.0

class TestGradientFreeShadowMode:
    def test_returns_production_output(self):
        from deployment import GradientFreeShadowMode
        prod = nn.Linear(8, 4)
        shadow = nn.Linear(8, 4)
        sm = GradientFreeShadowMode(prod, shadow, log_dir='/tmp/shadow_test')
        x = torch.randn(2, 8)
        out = sm(x)
        expected = prod(x)
        assert torch.allclose(out, expected)

    def test_discrepancy_tracking(self):
        from deployment import GradientFreeShadowMode
        prod = nn.Linear(8, 4)
        shadow = nn.Linear(8, 4)
        sm = GradientFreeShadowMode(prod, shadow, log_dir='/tmp/shadow_test2')
        for _ in range(5):
            sm(torch.randn(2, 8))
        stats = sm.discrepancy_stats()
        assert 'mean_discrepancy' in stats
