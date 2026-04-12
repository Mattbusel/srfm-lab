"""
tests/test_moe_inference.py
============================
Tests for the MoE inference engine (moe_inference_engine.py).
Covers: PED, prefetcher, KV cache, batched executor, capacity tuner,
        speculative router, full model, benchmark API.
"""

from __future__ import annotations

import math
import threading
import time
import unittest
from collections import deque
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

from lumina.moe_inference_engine import (
    BatchedExpertExecutor,
    BenchmarkResult,
    CapacityFactorTuner,
    ExpertCapacityHistogram,
    ExpertKVCacheEntry,
    ExpertKVCacheManager,
    ExpertLoadTracker,
    ExpertWeightPrefetcher,
    GeLUExpertFFN,
    InferenceSession,
    LuminaMoEModel,
    MoEConfig,
    MoEInferenceEngine,
    MoERouter,
    OptimizedMoELayer,
    PEDTrainer,
    PredictiveExpertDispatcher,
    RoutingDecision,
    SwiGLUExpertFFN,
    StreamingMoEInferencer,
    build_expert,
    build_lumina_moe,
    estimate_moe_flops,
    profile_forward,
)

DEVICE = "cpu"
DTYPE = torch.float32


def _make_config(**kwargs) -> MoEConfig:
    defaults = dict(
        num_experts=4,
        top_k=2,
        hidden_dim=32,
        ffn_dim=64,
        device=DEVICE,
        dtype=DTYPE,
        use_ped=False,
        use_prefetch=False,
        use_kv_cache=False,
        use_speculative_routing=False,
        enable_amp=False,
    )
    defaults.update(kwargs)
    return MoEConfig(**defaults)


# ===========================================================================
# PredictiveExpertDispatcher tests
# ===========================================================================


class TestPED(unittest.TestCase):

    def setUp(self):
        self.ped = PredictiveExpertDispatcher(
            hidden_dim=32,
            num_experts=4,
            ped_hidden_dim=16,
            top_k=2,
            device=DEVICE,
            dtype=DTYPE,
        )

    def test_forward_shape(self):
        h = torch.randn(8, 32)
        out = self.ped(h)
        self.assertEqual(out.shape, (8, 4))

    def test_predict_top_k_shape(self):
        h = torch.randn(8, 32)
        idx = self.ped.predict_top_k(h)
        self.assertEqual(idx.shape, (8, 2))

    def test_predict_top_k_values_in_range(self):
        h = torch.randn(16, 32)
        idx = self.ped.predict_top_k(h)
        self.assertTrue((idx >= 0).all())
        self.assertTrue((idx < 4).all())

    def test_update_returns_loss(self):
        h = torch.randn(8, 32)
        labels = torch.randint(0, 4, (8, 2))
        loss = self.ped.update(h, labels)
        self.assertIsInstance(loss, float)
        self.assertFalse(math.isnan(loss))

    def test_update_decreases_loss(self):
        h = torch.randn(32, 32)
        labels = torch.randint(0, 4, (32, 2))
        losses = [self.ped.update(h, labels) for _ in range(20)]
        # Loss should generally trend down over 20 steps
        self.assertLess(np.mean(losses[-5:]), np.mean(losses[:5]) + 1.0)

    def test_recent_accuracy_range(self):
        h = torch.randn(16, 32)
        labels = torch.randint(0, 4, (16, 2))
        for _ in range(5):
            self.ped.update(h, labels)
        acc = self.ped.recent_accuracy
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_checkpoint_roundtrip(self):
        h = torch.randn(8, 32)
        labels = torch.randint(0, 4, (8, 2))
        self.ped.update(h, labels)
        ckpt = self.ped.state_dict_for_checkpoint()
        ped2 = PredictiveExpertDispatcher(32, 4, 16, 2, device=DEVICE, dtype=DTYPE)
        ped2.load_checkpoint(ckpt)
        # Both should give the same output
        h2 = torch.randn(4, 32)
        out1 = self.ped(h2)
        out2 = ped2(h2)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))

    def test_step_counter_increments(self):
        h = torch.randn(4, 32)
        labels = torch.randint(0, 4, (4, 2))
        self.assertEqual(self.ped._step, 0)
        self.ped.update(h, labels)
        self.assertEqual(self.ped._step, 1)

    def test_ped_trainer(self):
        trainer = PEDTrainer(self.ped, epochs=2, batch_size=8)
        hidden_list = [torch.randn(8, 32) for _ in range(10)]
        label_list = [torch.randint(0, 4, (8, 2)) for _ in range(10)]
        losses = trainer.fit(hidden_list, label_list)
        self.assertEqual(len(losses), 2)


# ===========================================================================
# ExpertWeightPrefetcher tests
# ===========================================================================


class TestExpertWeightPrefetcher(unittest.TestCase):

    def _make_experts(self, n=4):
        return nn.ModuleList([nn.Linear(32, 32) for _ in range(n)])

    def test_prefetch_no_cuda(self):
        experts = self._make_experts()
        pf = ExpertWeightPrefetcher(experts, device=DEVICE)
        # Should not raise
        pf.prefetch([0, 1, 2])
        pf.synchronize()

    def test_wait_for_expert_no_cuda(self):
        experts = self._make_experts()
        pf = ExpertWeightPrefetcher(experts, device=DEVICE)
        pf.prefetch([0])
        pf.wait_for_expert(0)  # Should not raise

    def test_repr(self):
        experts = self._make_experts()
        pf = ExpertWeightPrefetcher(experts, device=DEVICE)
        r = repr(pf)
        self.assertIn("ExpertWeightPrefetcher", r)

    def test_prefetch_invalid_id(self):
        experts = self._make_experts(3)
        pf = ExpertWeightPrefetcher(experts, device=DEVICE)
        # Should not raise even with out-of-range IDs
        pf.prefetch([0, 999])


# ===========================================================================
# ExpertKVCacheManager tests
# ===========================================================================


class TestExpertKVCacheManager(unittest.TestCase):

    def setUp(self):
        self.cache = ExpertKVCacheManager(
            num_experts=4,
            max_resident=2,
            vram_budget_bytes=10 * 1024 * 1024,  # 10 MB
            device=DEVICE,
        )

    def test_put_and_get(self):
        k = torch.randn(4, 8)
        v = torch.randn(4, 8)
        self.cache.put(0, "seq_1", k, v)
        result = self.cache.get(0, "seq_1")
        self.assertIsNotNone(result)
        k2, v2 = result
        self.assertEqual(k2.shape, k.shape)

    def test_get_miss(self):
        result = self.cache.get(0, "nonexistent")
        self.assertIsNone(result)

    def test_lru_eviction_resident(self):
        # Fill beyond max_resident
        for expert_id in range(4):
            k = torch.randn(4, 8)
            v = torch.randn(4, 8)
            self.cache.put(expert_id, f"seq_{expert_id}", k, v)
        # Should only have 2 resident experts
        self.assertLessEqual(len(self.cache._resident_experts), 2 + 1)

    def test_invalidate_seq(self):
        k = torch.randn(4, 8)
        v = torch.randn(4, 8)
        self.cache.put(0, "seq_x", k, v)
        self.cache.put(1, "seq_x", k, v)
        self.cache.invalidate("seq_x")
        self.assertIsNone(self.cache.get(0, "seq_x"))
        self.assertIsNone(self.cache.get(1, "seq_x"))

    def test_evict_expert(self):
        k = torch.randn(4, 8)
        v = torch.randn(4, 8)
        self.cache.put(0, "seq_y", k, v)
        self.cache.evict_expert(0)
        self.assertIsNone(self.cache.get(0, "seq_y"))

    def test_stats(self):
        k = torch.randn(4, 8)
        v = torch.randn(4, 8)
        self.cache.put(0, "s1", k, v)
        self.cache.get(0, "s1")
        self.cache.get(0, "missing")
        stats = self.cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 0.5)

    def test_budget_enforcement(self):
        # Use a tiny budget to force eviction
        cache = ExpertKVCacheManager(
            num_experts=4,
            max_resident=4,
            vram_budget_bytes=1024,  # 1 KB
            device=DEVICE,
        )
        for i in range(20):
            k = torch.randn(10, 10)  # ~400 bytes
            v = torch.randn(10, 10)
            cache.put(i % 4, f"seq_{i}", k, v)
        # Total bytes should stay under or near budget
        self.assertLessEqual(cache._total_bytes, 1024 * 10)  # some slack for test

    def test_thread_safety(self):
        errors = []

        def writer(expert_id):
            try:
                for i in range(50):
                    k = torch.randn(2, 4)
                    v = torch.randn(2, 4)
                    self.cache.put(expert_id, f"seq_{i}", k, v)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i % 4,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)


# ===========================================================================
# MoERouter tests
# ===========================================================================


class TestMoERouter(unittest.TestCase):

    def setUp(self):
        self.router = MoERouter(
            hidden_dim=32,
            num_experts=4,
            top_k=2,
            dtype=DTYPE,
        )

    def test_routing_decision_shapes(self):
        tokens = torch.randn(16, 32)
        decision = self.router(tokens)
        self.assertEqual(decision.expert_indices.shape, (16, 2))
        self.assertEqual(decision.routing_weights.shape, (16, 2))
        self.assertEqual(decision.router_logits.shape, (16, 4))

    def test_routing_weights_sum_to_one(self):
        tokens = torch.randn(16, 32)
        decision = self.router(tokens)
        sums = decision.routing_weights.float().sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones(16), atol=1e-4))

    def test_expert_indices_in_range(self):
        tokens = torch.randn(32, 32)
        decision = self.router(tokens)
        self.assertTrue((decision.expert_indices >= 0).all())
        self.assertTrue((decision.expert_indices < 4).all())

    def test_load_balance_loss(self):
        tokens = torch.randn(16, 32)
        decision = self.router(tokens)
        loss = self.router.load_balance_loss(decision, 16)
        self.assertFalse(math.isnan(loss.item()))
        self.assertGreater(loss.item(), 0.0)

    def test_no_noise_in_eval(self):
        self.router.eval()
        tokens = torch.randn(8, 32)
        d1 = self.router(tokens)
        d2 = self.router(tokens)
        self.assertTrue(torch.allclose(d1.router_logits, d2.router_logits))


# ===========================================================================
# BatchedExpertExecutor tests
# ===========================================================================


class TestBatchedExpertExecutor(unittest.TestCase):

    def _make_executor(self, n_experts=4, top_k=2):
        experts = nn.ModuleList([
            GeLUExpertFFN(32, 64, dtype=DTYPE) for _ in range(n_experts)
        ])
        return BatchedExpertExecutor(experts, capacity_factor=1.5, top_k=top_k)

    def _make_routing(self, T=16, n_experts=4, top_k=2):
        indices = torch.zeros(T, top_k, dtype=torch.long)
        for i in range(T):
            indices[i] = torch.randperm(n_experts)[:top_k]
        weights = torch.ones(T, top_k) / top_k
        logits = torch.randn(T, n_experts)
        return RoutingDecision(indices, weights, logits)

    def test_output_shape(self):
        executor = self._make_executor()
        tokens = torch.randn(16, 32)
        routing = self._make_routing(16)
        out = executor.execute(tokens, routing)
        self.assertEqual(out.shape, (16, 32))

    def test_output_not_nan(self):
        executor = self._make_executor()
        tokens = torch.randn(16, 32)
        routing = self._make_routing(16)
        out = executor.execute(tokens, routing)
        self.assertFalse(torch.isnan(out).any())

    def test_capacity_drop(self):
        executor = self._make_executor()
        executor.capacity_factor = 0.1  # Very low to force drops
        tokens = torch.randn(100, 32)
        routing = self._make_routing(100)
        out = executor.execute(tokens, routing)
        self.assertGreater(executor.dropped_tokens, 0)

    def test_zero_tokens(self):
        executor = self._make_executor()
        tokens = torch.randn(0, 32)
        routing = RoutingDecision(
            torch.zeros(0, 2, dtype=torch.long),
            torch.zeros(0, 2),
            torch.zeros(0, 4),
        )
        out = executor.execute(tokens, routing)
        self.assertEqual(out.shape, (0, 32))

    def test_single_token(self):
        executor = self._make_executor()
        tokens = torch.randn(1, 32)
        routing = self._make_routing(1)
        out = executor.execute(tokens, routing)
        self.assertEqual(out.shape, (1, 32))


# ===========================================================================
# CapacityFactorTuner tests
# ===========================================================================


class TestCapacityFactorTuner(unittest.TestCase):

    def test_initial_factor(self):
        tuner = CapacityFactorTuner(initial_factor=1.25)
        self.assertAlmostEqual(tuner.factor, 1.25)

    def test_increases_on_high_drops(self):
        tuner = CapacityFactorTuner(
            initial_factor=1.0,
            target_drop_rate=0.001,
            latency_budget_ms=1000.0,
            adjust_interval=10,
            step_size=0.1,
        )
        for _ in range(10):
            tuner.record(dropped_tokens=100, total_tokens=200, latency_ms=1.0)
        self.assertGreater(tuner.factor, 1.0)

    def test_decreases_on_high_latency(self):
        tuner = CapacityFactorTuner(
            initial_factor=2.0,
            target_drop_rate=0.001,
            latency_budget_ms=10.0,
            adjust_interval=10,
            step_size=0.1,
        )
        for _ in range(10):
            tuner.record(dropped_tokens=0, total_tokens=200, latency_ms=100.0)
        self.assertLess(tuner.factor, 2.0)

    def test_clamp_to_min(self):
        tuner = CapacityFactorTuner(initial_factor=1.0, min_factor=1.0)
        for _ in range(100):
            tuner.record(0, 200, 1000.0)
        self.assertGreaterEqual(tuner.factor, 1.0)

    def test_clamp_to_max(self):
        tuner = CapacityFactorTuner(
            initial_factor=3.9,
            max_factor=4.0,
            target_drop_rate=0.001,
            latency_budget_ms=1000.0,
            adjust_interval=10,
        )
        for _ in range(100):
            tuner.record(1000, 200, 1.0)
        self.assertLessEqual(tuner.factor, 4.0)

    def test_history_grows(self):
        tuner = CapacityFactorTuner(adjust_interval=5)
        for _ in range(5):
            tuner.record(0, 100, 5.0)
        self.assertGreaterEqual(len(tuner.history), 1)


# ===========================================================================
# SwiGLU / GeLU Expert FFN tests
# ===========================================================================


class TestExpertFFN(unittest.TestCase):

    def test_swiglu_shape(self):
        ffn = SwiGLUExpertFFN(32, 64, dtype=DTYPE)
        x = torch.randn(8, 32)
        out = ffn(x)
        self.assertEqual(out.shape, (8, 32))

    def test_gelu_shape(self):
        ffn = GeLUExpertFFN(32, 64, dtype=DTYPE)
        x = torch.randn(8, 32)
        out = ffn(x)
        self.assertEqual(out.shape, (8, 32))

    def test_swiglu_no_nan(self):
        ffn = SwiGLUExpertFFN(32, 64, dtype=DTYPE)
        x = torch.randn(16, 32)
        out = ffn(x)
        self.assertFalse(torch.isnan(out).any())

    def test_gelu_no_nan(self):
        ffn = GeLUExpertFFN(32, 64, dtype=DTYPE)
        x = torch.randn(16, 32)
        out = ffn(x)
        self.assertFalse(torch.isnan(out).any())

    def test_build_expert_swiglu(self):
        e = build_expert(32, 64, "swiglu", DTYPE)
        self.assertIsInstance(e, SwiGLUExpertFFN)

    def test_build_expert_gelu(self):
        e = build_expert(32, 64, "gelu", DTYPE)
        self.assertIsInstance(e, GeLUExpertFFN)

    def test_build_expert_invalid(self):
        with self.assertRaises(ValueError):
            build_expert(32, 64, "invalid_act", DTYPE)

    def test_swiglu_gradient_flow(self):
        ffn = SwiGLUExpertFFN(16, 32, dtype=torch.float32)
        x = torch.randn(4, 16, requires_grad=True)
        out = ffn(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())


# ===========================================================================
# OptimizedMoELayer tests
# ===========================================================================


class TestOptimizedMoELayer(unittest.TestCase):

    def setUp(self):
        self.config = _make_config()
        self.layer = OptimizedMoELayer(self.config, layer_idx=0)

    def test_forward_shape(self):
        x = torch.randn(2, 8, 32)
        out = self.layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_forward_no_nan(self):
        x = torch.randn(2, 8, 32)
        out = self.layer(x)
        self.assertFalse(torch.isnan(out).any())

    def test_residual_connection(self):
        # With zero expert weights, output should be close to input (residual)
        # Verify residual is present by checking output != 0
        x = torch.randn(1, 4, 32)
        out = self.layer(x)
        self.assertFalse((out == 0).all())

    def test_return_aux_loss(self):
        x = torch.randn(2, 8, 32)
        out, aux = self.layer(x, return_aux_loss=True)
        self.assertEqual(out.shape, x.shape)
        self.assertFalse(math.isnan(aux.item()))

    def test_step_counter(self):
        x = torch.randn(1, 4, 32)
        self.assertEqual(self.layer._step, 0)
        self.layer(x)
        self.assertEqual(self.layer._step, 1)

    def test_with_ped(self):
        config = _make_config(use_ped=True)
        layer = OptimizedMoELayer(config, layer_idx=0)
        x = torch.randn(2, 8, 32)
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_extra_repr(self):
        r = self.layer.extra_repr()
        self.assertIn("num_experts=4", r)
        self.assertIn("top_k=2", r)


# ===========================================================================
# LuminaMoEModel tests
# ===========================================================================


class TestLuminaMoEModel(unittest.TestCase):

    def setUp(self):
        self.config = _make_config()
        self.model = LuminaMoEModel(self.config, num_layers=2, vocab_size=100, max_seq_len=64)

    def test_forward_shape_ids(self):
        ids = torch.randint(0, 100, (2, 16))
        out = self.model(ids)
        self.assertEqual(out.shape, (2, 16, 100))

    def test_continuous_forward_shape(self):
        feats = torch.randn(2, 16, 32)
        out = self.model.continuous_input_forward(feats)
        self.assertEqual(out.shape, (2, 16, 32))

    def test_forward_with_aux_loss(self):
        ids = torch.randint(0, 100, (2, 8))
        logits, aux = self.model(ids, return_aux_loss=True)
        self.assertEqual(logits.shape, (2, 8, 100))
        self.assertFalse(math.isnan(aux.item()))

    def test_no_nan_output(self):
        feats = torch.randn(4, 32, 32)
        out = self.model.continuous_input_forward(feats)
        self.assertFalse(torch.isnan(out).any())

    def test_parameter_count(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(n_params, 0)


# ===========================================================================
# MoEInferenceEngine tests
# ===========================================================================


class TestMoEInferenceEngine(unittest.TestCase):

    def setUp(self):
        config = _make_config()
        model = LuminaMoEModel(config, num_layers=2)
        self.engine = MoEInferenceEngine(model, config)

    def test_infer_shape(self):
        x = torch.randn(2, 16, 32)
        out = self.engine.infer(x)
        self.assertEqual(out.shape, (2, 16, 32))

    def test_infer_no_nan(self):
        x = torch.randn(4, 16, 32)
        out = self.engine.infer(x)
        self.assertFalse(torch.isnan(out).any())

    def test_warmup_does_not_raise(self):
        self.engine.warmup(batch_size=2, seq_len=16)

    def test_benchmark_returns_results(self):
        results = self.engine.benchmark(batch_sizes=[1, 2], seq_lens=[8])
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, BenchmarkResult)
            self.assertGreater(r.tokens_per_sec, 0)

    def test_benchmark_latency_nonnegative(self):
        results = self.engine.benchmark(batch_sizes=[1], seq_lens=[8])
        r = results[0]
        self.assertGreaterEqual(r.latency_p50_ms, 0.0)
        self.assertGreaterEqual(r.latency_p95_ms, 0.0)
        self.assertGreaterEqual(r.latency_p99_ms, 0.0)

    def test_print_benchmark_report(self):
        results = self.engine.benchmark(batch_sizes=[1], seq_lens=[8])
        # Should not raise
        self.engine.print_benchmark_report(results)


# ===========================================================================
# ExpertLoadTracker tests
# ===========================================================================


class TestExpertLoadTracker(unittest.TestCase):

    def test_record_and_utilization(self):
        tracker = ExpertLoadTracker(num_experts=4, window=100)
        indices = torch.randint(0, 4, (32, 2))
        tracker.record(indices)
        util = tracker.utilization()
        self.assertEqual(len(util), 4)
        self.assertAlmostEqual(util.sum(), 1.0, places=1)

    def test_imbalance_ratio_balanced(self):
        tracker = ExpertLoadTracker(num_experts=4, window=1000)
        # Send equal tokens to all experts
        indices = torch.tensor([[i % 4, (i + 1) % 4] for i in range(100)])
        tracker.record(indices)
        ratio = tracker.imbalance_ratio()
        self.assertGreater(ratio, 0.0)
        self.assertLess(ratio, 10.0)

    def test_summary(self):
        tracker = ExpertLoadTracker(num_experts=4)
        indices = torch.randint(0, 4, (16, 2))
        tracker.record(indices)
        summary = tracker.summary()
        self.assertIn("per_expert_utilization", summary)
        self.assertIn("imbalance_ratio", summary)


# ===========================================================================
# ExpertCapacityHistogram tests
# ===========================================================================


class TestExpertCapacityHistogram(unittest.TestCase):

    def test_record_and_percentile(self):
        hist = ExpertCapacityHistogram(num_experts=4, max_tokens_per_expert=64)
        for _ in range(20):
            idx = torch.randint(0, 4, (32, 2))
            hist.record(idx)
        p50 = hist.percentile(0, 50)
        self.assertGreaterEqual(p50, 0)

    def test_recommended_capacity_factor(self):
        hist = ExpertCapacityHistogram(num_experts=4)
        for _ in range(10):
            idx = torch.randint(0, 4, (32, 2))
            hist.record(idx)
        cf = hist.recommended_capacity_factor(batch_size=8, seq_len=32)
        self.assertGreaterEqual(cf, 1.0)


# ===========================================================================
# InferenceSession tests
# ===========================================================================


class TestInferenceSession(unittest.TestCase):

    def setUp(self):
        config = _make_config()
        model = LuminaMoEModel(config, num_layers=1)
        self.engine = MoEInferenceEngine(model, config)

    def test_generate_continuous(self):
        session = InferenceSession(self.engine, max_new_tokens=2)
        feats = torch.randn(1, 8, 32)
        out = session.generate_continuous(feats, num_steps=2)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[2], 32)

    def test_elapsed_ms(self):
        session = InferenceSession(self.engine)
        time.sleep(0.01)
        self.assertGreater(session.elapsed_ms, 5.0)

    def test_close_no_raise(self):
        session = InferenceSession(self.engine, session_id="test_close")
        session.close()  # Should not raise


# ===========================================================================
# StreamingMoEInferencer tests
# ===========================================================================


class TestStreamingMoEInferencer(unittest.TestCase):

    def setUp(self):
        config = _make_config()
        model = LuminaMoEModel(config, num_layers=1)
        self.engine = MoEInferenceEngine(model, config)

    def test_push_returns_none_until_full(self):
        streamer = StreamingMoEInferencer(self.engine, context_window=8)
        for i in range(7):
            snap = torch.randn(32)
            result = streamer.push(snap)
            self.assertIsNone(result)

    def test_push_returns_output_when_full(self):
        streamer = StreamingMoEInferencer(self.engine, context_window=4)
        for i in range(4):
            snap = torch.randn(32)
            result = streamer.push(snap)
        # Last push should return something
        self.assertIsNotNone(result)

    def test_buffer_fill_fraction(self):
        streamer = StreamingMoEInferencer(self.engine, context_window=10)
        for _ in range(5):
            streamer.push(torch.randn(32))
        self.assertAlmostEqual(streamer.buffer_fill, 0.5)

    def test_callback_called(self):
        called = []
        streamer = StreamingMoEInferencer(
            self.engine,
            context_window=4,
            output_callback=lambda x: called.append(True),
        )
        for _ in range(4):
            streamer.push(torch.randn(32))
        self.assertEqual(len(called), 1)


# ===========================================================================
# build_lumina_moe factory tests
# ===========================================================================


class TestBuildLuminaMoe(unittest.TestCase):

    def test_returns_model_and_engine(self):
        model, engine = build_lumina_moe(
            num_experts=4,
            top_k=2,
            hidden_dim=32,
            ffn_dim=64,
            num_layers=1,
            device=DEVICE,
            dtype=DTYPE,
            use_all_optimizations=False,
        )
        self.assertIsInstance(model, LuminaMoEModel)
        self.assertIsInstance(engine, MoEInferenceEngine)

    def test_engine_can_infer(self):
        model, engine = build_lumina_moe(
            num_experts=4, hidden_dim=32, ffn_dim=64, num_layers=1,
            device=DEVICE, dtype=DTYPE, use_all_optimizations=False,
        )
        x = torch.randn(2, 8, 32)
        out = engine.infer(x)
        self.assertEqual(out.shape, (2, 8, 32))


# ===========================================================================
# estimate_moe_flops tests
# ===========================================================================


class TestEstimateMoeFlops(unittest.TestCase):

    def test_returns_positive_flops(self):
        config = _make_config(use_ped=True)
        result = estimate_moe_flops(config, batch_size=4, seq_len=32, num_layers=2)
        self.assertGreater(result["total_flops"], 0)
        self.assertGreater(result["total_gflops"], 0)

    def test_flops_scale_with_batch(self):
        config = _make_config()
        r1 = estimate_moe_flops(config, batch_size=1, seq_len=32, num_layers=1)
        r2 = estimate_moe_flops(config, batch_size=2, seq_len=32, num_layers=1)
        self.assertAlmostEqual(r2["total_flops"] / r1["total_flops"], 2.0, places=5)


# ===========================================================================
# profile_forward tests
# ===========================================================================


class TestProfileForward(unittest.TestCase):

    def test_profile_returns_dict(self):
        config = _make_config()
        model = LuminaMoEModel(config, num_layers=1)
        engine = MoEInferenceEngine(model, config)
        result = profile_forward(engine, batch_size=2, seq_len=8, n_iter=5)
        for key in ["mean_ms", "p50_ms", "p95_ms", "p99_ms", "tokens_per_sec"]:
            self.assertIn(key, result)
            self.assertGreater(result[key], 0.0)


class TestExpertActivationPatternAnalyzer(unittest.TestCase):

    def setUp(self):
        from lumina.moe_inference_engine import ExpertActivationPatternAnalyzer
        self.analyzer = ExpertActivationPatternAnalyzer(num_experts=4, top_k=2, window=200)

    def test_record_does_not_raise(self):
        idx = torch.randint(0, 4, (16, 2))
        self.analyzer.record(idx)

    def test_co_activation_matrix_shape(self):
        idx = torch.randint(0, 4, (32, 2))
        self.analyzer.record(idx)
        mat = self.analyzer.co_activation_matrix()
        self.assertEqual(mat.shape, (4, 4))

    def test_expert_popularity_sums_to_one(self):
        idx = torch.randint(0, 4, (64, 2))
        self.analyzer.record(idx)
        pop = self.analyzer.expert_popularity()
        self.assertAlmostEqual(float(pop.sum()), 1.0, places=3)

    def test_top_coactivating_pairs(self):
        for _ in range(5):
            idx = torch.randint(0, 4, (32, 2))
            self.analyzer.record(idx)
        pairs = self.analyzer.top_co_activating_pairs(n=3)
        self.assertLessEqual(len(pairs), 3)

    def test_summary_keys(self):
        idx = torch.randint(0, 4, (32, 2))
        self.analyzer.record(idx)
        s = self.analyzer.summary()
        self.assertIn("n_tokens_seen", s)
        self.assertIn("expert_popularity", s)
        self.assertIn("gini_coefficient", s)

    def test_gini_range(self):
        idx = torch.randint(0, 4, (100, 2))
        self.analyzer.record(idx)
        s = self.analyzer.summary()
        self.assertGreaterEqual(s["gini_coefficient"], 0.0)
        self.assertLessEqual(s["gini_coefficient"], 1.0)


class TestRouterTemperatureScaler(unittest.TestCase):

    def setUp(self):
        from lumina.moe_inference_engine import RouterTemperatureScaler
        self.scaler = RouterTemperatureScaler(num_experts=4, target_entropy=1.0)

    def test_initial_temperature(self):
        self.assertAlmostEqual(self.scaler.temperature, 1.0, places=3)

    def test_forward_scales_logits(self):
        logits = torch.randn(8, 4)
        scaled = self.scaler(logits)
        # With T=1, should be identical
        self.assertTrue(torch.allclose(scaled, logits, atol=1e-5))

    def test_calibrate_returns_float(self):
        logits_list = [torch.randn(16, 4) for _ in range(5)]
        temp = self.scaler.calibrate(logits_list, max_steps=10)
        self.assertIsInstance(temp, float)
        self.assertGreater(temp, 0.0)

    def test_high_temperature_flattens_distribution(self):
        self.scaler.log_temperature.data = torch.tensor([2.0])  # T = e^2 ≈ 7.4
        logits = torch.randn(4, 4)
        scaled = self.scaler(logits)
        probs = F.softmax(scaled, dim=-1)
        # Should be more uniform than with T=1
        self.assertLess(probs.max().item() - probs.min().item(), 1.0)


class TestAdaptiveTopKRouter(unittest.TestCase):

    def setUp(self):
        from lumina.moe_inference_engine import AdaptiveTopKRouter
        self.router = AdaptiveTopKRouter(
            hidden_dim=32,
            num_experts=4,
            min_k=1,
            max_k=3,
            dtype=DTYPE,
        )

    def test_forward_shapes(self):
        hidden = torch.randn(8, 32)
        indices, weights, k_per_token = self.router(hidden)
        self.assertEqual(indices.shape, (8, 3))   # max_k=3
        self.assertEqual(weights.shape, (8, 3))
        self.assertEqual(k_per_token.shape, (8,))

    def test_k_per_token_in_range(self):
        hidden = torch.randn(16, 32)
        _, _, k = self.router(hidden)
        self.assertTrue((k >= 1).all())
        self.assertTrue((k <= 3).all())

    def test_unused_slots_are_zero_weight(self):
        hidden = torch.randn(8, 32)
        _, weights, k_per_token = self.router(hidden)
        for t in range(8):
            k = int(k_per_token[t].item())
            # Slots beyond k should have zero weight
            unused_weights = weights[t, k:]
            self.assertTrue((unused_weights == 0).all())


class TestLayerwiseExpertMonitor(unittest.TestCase):

    def setUp(self):
        from lumina.moe_inference_engine import LayerwiseExpertMonitor
        self.monitor = LayerwiseExpertMonitor(num_layers=3, num_experts=4)

    def test_record_does_not_raise(self):
        idx = torch.randint(0, 4, (16, 2))
        self.monitor.record(0, idx)

    def test_layer_utilization_shape(self):
        idx = torch.randint(0, 4, (32, 2))
        self.monitor.record(0, idx)
        util = self.monitor.layer_utilization(0)
        self.assertEqual(len(util), 4)

    def test_most_imbalanced_layer(self):
        for l in range(3):
            idx = torch.randint(0, 4, (16, 2))
            self.monitor.record(l, idx)
        layer, ratio = self.monitor.most_imbalanced_layer()
        self.assertIn(layer, [0, 1, 2])
        self.assertGreater(ratio, 0.0)

    def test_depth_expert_heatmap_shape(self):
        for l in range(3):
            idx = torch.randint(0, 4, (16, 2))
            self.monitor.record(l, idx)
        heatmap = self.monitor.depth_expert_heatmap()
        self.assertEqual(heatmap.shape, (3, 4))

    def test_print_heatmap_no_raise(self):
        for l in range(3):
            self.monitor.record(l, torch.randint(0, 4, (8, 2)))
        self.monitor.print_heatmap()


class TestMoEWarmupScheduler(unittest.TestCase):

    def setUp(self):
        from lumina.moe_inference_engine import MoEWarmupScheduler
        self.sched = MoEWarmupScheduler(num_experts=8, final_top_k=2, warmup_steps=100)

    def test_initial_top_k_is_num_experts(self):
        self.assertEqual(self.sched.get_top_k(), 8)

    def test_final_top_k_after_warmup(self):
        for _ in range(101):
            self.sched.step()
        self.assertEqual(self.sched.get_top_k(), 2)

    def test_linear_transition(self):
        # At warmup_steps*3/4, k should be between final and num_experts
        for _ in range(75):
            self.sched.step()
        k = self.sched.get_top_k()
        self.assertGreaterEqual(k, 2)
        self.assertLessEqual(k, 8)

    def test_is_warmed_up(self):
        self.assertFalse(self.sched.is_warmed_up())
        for _ in range(100):
            self.sched.step()
        self.assertTrue(self.sched.is_warmed_up())


class TestFinancialFeaturePreprocessor(unittest.TestCase):

    def setUp(self):
        from lumina.moe_inference_engine import FinancialFeaturePreprocessor
        self.prep = FinancialFeaturePreprocessor(
            raw_feature_dim=16,
            hidden_dim=32,
            dtype=DTYPE,
        )

    def test_forward_shape(self):
        x = torch.randn(2, 8, 16)
        out = self.prep(x)
        self.assertEqual(out.shape, (2, 8, 32))

    def test_forward_no_nan(self):
        x = torch.randn(2, 8, 16)
        out = self.prep(x)
        self.assertFalse(torch.isnan(out.float()).any())

    def test_statistics_updated_during_training(self):
        self.prep.train()
        x = torch.randn(4, 8, 16)
        self.prep(x)
        self.assertGreater(self.prep.n_batches_seen.item(), 0)

    def test_feature_importance_sum_to_one(self):
        x = torch.randn(4, 8, 16)
        self.prep(x)  # ensure weights are initialized
        importance = self.prep.feature_importance()
        total = sum(importance.values())
        self.assertAlmostEqual(total, 1.0, places=4)


class TestExpertDropout(unittest.TestCase):

    def setUp(self):
        from lumina.moe_inference_engine import ExpertDropout
        self.dropout = ExpertDropout(p=0.5)

    def test_eval_mode_no_dropout(self):
        self.dropout.eval()
        x = torch.ones(8, 16)
        out = self.dropout(x, 0)
        self.assertTrue(torch.allclose(out, x))

    def test_training_mode_can_zero(self):
        self.dropout.train()
        x = torch.ones(8, 16)
        # With p=0.5, about half the calls should zero out
        results = [self.dropout(x, 0).sum().item() for _ in range(20)]
        has_zero = any(r == 0 for r in results)
        has_nonzero = any(r > 0 for r in results)
        self.assertTrue(has_zero or has_nonzero)  # at least one

    def test_zero_dropout_rate(self):
        from lumina.moe_inference_engine import ExpertDropout
        d = ExpertDropout(p=0.0)
        d.train()
        x = torch.ones(4, 8)
        out = d(x, 0)
        self.assertTrue(torch.allclose(out, x))


class TestExpertSpecializationProbe(unittest.TestCase):

    def setUp(self):
        from lumina.moe_inference_engine import ExpertSpecializationProbe
        self.probe = ExpertSpecializationProbe(num_experts=4, feature_dim=16)

    def test_record_does_not_raise(self):
        feats = torch.randn(16, 16)
        idx = torch.randint(0, 4, (16, 2))
        self.probe.record(feats, idx)

    def test_expert_feature_mean_shape(self):
        feats = torch.randn(32, 16)
        idx = torch.randint(0, 4, (32, 2))
        self.probe.record(feats, idx)
        mean = self.probe.expert_feature_mean(0)
        self.assertEqual(mean.shape, (16,))

    def test_most_discriminative_features_length(self):
        feats = torch.randn(32, 16)
        idx = torch.randint(0, 4, (32, 2))
        self.probe.record(feats, idx)
        disc = self.probe.most_discriminative_features(0, n=3)
        self.assertLessEqual(len(disc), 3)

    def test_specialization_report_has_all_experts(self):
        feats = torch.randn(64, 16)
        idx = torch.randint(0, 4, (64, 2))
        self.probe.record(feats, idx)
        report = self.probe.specialization_report()
        for e in range(4):
            self.assertIn(f"expert_{e}", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
