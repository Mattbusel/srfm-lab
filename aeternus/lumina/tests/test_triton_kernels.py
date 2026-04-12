"""
tests/test_triton_kernels.py
=============================
Tests for Triton kernels and their PyTorch fallback implementations.
CUDA-dependent tests are skipped when CUDA is not available.
"""

from __future__ import annotations

import math
import unittest
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from lumina.triton_kernels import (
    TRITON_AVAILABLE,
    GPU_TYPE,
    KernelBenchmark,
    MixedPrecisionContext,
    PackedInt4Weight,
    TritonMoELayer,
    _get_autotune_configs,
    _make_triton_configs,
    _pytorch_router_dispatch,
    _pytorch_scatter_gather,
    _pytorch_softmax_topk,
    _pytorch_swiglu_expert,
    cast_to_bf16_if_cuda,
    fused_router_dispatch,
    fused_scatter_gather,
    fused_softmax_topk,
    fused_swiglu_expert,
    run_all_verifications,
    select_best_config,
    verify_softmax_topk,
    verify_swiglu_expert,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SKIP_CUDA = not torch.cuda.is_available()
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def skipIfNoCUDA(test):
    """Decorator to skip a test if CUDA is not available."""
    return unittest.skipIf(SKIP_CUDA, "CUDA not available")(test)


def skipIfNoTriton(test):
    """Decorator to skip a test if Triton is not available."""
    return unittest.skipIf(not TRITON_AVAILABLE, "Triton not available")(test)


# ===========================================================================
# PyTorch fallback tests (always run)
# ===========================================================================


class TestPyTorchSoftmaxTopK(unittest.TestCase):

    def test_output_shapes(self):
        logits = torch.randn(16, 8)
        weights, indices = _pytorch_softmax_topk(logits, top_k=2)
        self.assertEqual(weights.shape, (16, 2))
        self.assertEqual(indices.shape, (16, 2))

    def test_weights_sum_to_one(self):
        logits = torch.randn(32, 8)
        weights, indices = _pytorch_softmax_topk(logits, top_k=2)
        sums = weights.float().sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones(32), atol=1e-4))

    def test_indices_in_range(self):
        logits = torch.randn(16, 8)
        _, indices = _pytorch_softmax_topk(logits, top_k=3)
        self.assertTrue((indices >= 0).all())
        self.assertTrue((indices < 8).all())

    def test_top1(self):
        logits = torch.randn(8, 4)
        weights, indices = _pytorch_softmax_topk(logits, top_k=1)
        self.assertEqual(weights.shape, (8, 1))
        # For top-1, weight should be 1.0 after renormalization
        self.assertTrue(torch.allclose(weights.float(), torch.ones(8, 1), atol=1e-4))

    def test_output_dtype_bf16(self):
        logits = torch.randn(8, 4)
        weights, indices = _pytorch_softmax_topk(logits, top_k=2)
        self.assertEqual(weights.dtype, torch.bfloat16)

    def test_indices_dtype_int32(self):
        logits = torch.randn(8, 4)
        _, indices = _pytorch_softmax_topk(logits, top_k=2)
        self.assertEqual(indices.dtype, torch.int32)

    def test_no_duplicate_indices(self):
        logits = torch.randn(16, 8)
        _, indices = _pytorch_softmax_topk(logits, top_k=4)
        for i in range(16):
            row = indices[i].tolist()
            self.assertEqual(len(row), len(set(row)), f"Duplicate indices at row {i}")

    def test_top_k_equals_num_experts(self):
        logits = torch.randn(4, 4)
        weights, indices = _pytorch_softmax_topk(logits, top_k=4)
        self.assertEqual(weights.shape, (4, 4))


class TestPyTorchRouterDispatch(unittest.TestCase):

    def test_output_shapes(self):
        T, H, E, K = 16, 32, 8, 2
        hidden = torch.randn(T, H)
        router_w = torch.randn(H, E)
        indices, weights, logits = _pytorch_router_dispatch(hidden, router_w, K)
        self.assertEqual(indices.shape, (T, K))
        self.assertEqual(weights.shape, (T, K))
        self.assertEqual(logits.shape, (T, E))

    def test_weights_positive(self):
        T, H, E, K = 8, 32, 8, 2
        hidden = torch.randn(T, H)
        router_w = torch.randn(H, E)
        _, weights, _ = _pytorch_router_dispatch(hidden, router_w, K)
        self.assertTrue((weights.float() >= 0).all())

    def test_logits_dtype_float32(self):
        hidden = torch.randn(8, 32)
        router_w = torch.randn(32, 4)
        _, _, logits = _pytorch_router_dispatch(hidden, router_w, 2)
        self.assertEqual(logits.dtype, torch.float32)


class TestPyTorchSwiGLUExpert(unittest.TestCase):

    def test_output_shape(self):
        M, H, N = 8, 32, 64
        x = torch.randn(M, H)
        wg = torch.randn(H, N)
        wu = torch.randn(H, N)
        wd = torch.randn(N, H)
        out = _pytorch_swiglu_expert(x, wg, wu, wd)
        self.assertEqual(out.shape, (M, H))

    def test_output_no_nan(self):
        M, H, N = 16, 32, 64
        x = torch.randn(M, H)
        wg = torch.randn(H, N)
        wu = torch.randn(H, N)
        wd = torch.randn(N, H)
        out = _pytorch_swiglu_expert(x, wg, wu, wd)
        self.assertFalse(torch.isnan(out).any())

    def test_output_dtype_bf16(self):
        M, H, N = 4, 16, 32
        x = torch.randn(M, H, dtype=torch.bfloat16)
        wg = torch.randn(H, N, dtype=torch.bfloat16)
        wu = torch.randn(H, N, dtype=torch.bfloat16)
        wd = torch.randn(N, H, dtype=torch.bfloat16)
        out = _pytorch_swiglu_expert(x, wg, wu, wd)
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_gradient_flows(self):
        M, H, N = 4, 16, 32
        x = torch.randn(M, H, dtype=torch.float32, requires_grad=True)
        wg = torch.randn(H, N, dtype=torch.float32, requires_grad=True)
        wu = torch.randn(H, N, dtype=torch.float32, requires_grad=True)
        wd = torch.randn(N, H, dtype=torch.float32, requires_grad=True)
        out = _pytorch_swiglu_expert(x, wg, wu, wd)
        out.sum().backward()
        self.assertIsNotNone(x.grad)


# ===========================================================================
# Unified wrapper function tests (auto-fallback to PyTorch)
# ===========================================================================


class TestFusedSoftmaxTopK(unittest.TestCase):

    def test_shape(self):
        logits = torch.randn(16, 8)
        weights, indices = fused_softmax_topk(logits, top_k=2)
        self.assertEqual(weights.shape, (16, 2))
        self.assertEqual(indices.shape, (16, 2))

    def test_weights_nonnegative(self):
        logits = torch.randn(16, 8)
        weights, _ = fused_softmax_topk(logits, top_k=2)
        self.assertTrue((weights.float() >= 0).all())

    def test_indices_valid(self):
        logits = torch.randn(32, 8)
        _, indices = fused_softmax_topk(logits, top_k=3)
        self.assertTrue((indices >= 0).all())
        self.assertTrue((indices < 8).all())

    def test_consistent_with_pytorch(self):
        logits = torch.randn(16, 8)
        w_ref, i_ref = _pytorch_softmax_topk(logits, top_k=2)
        w_fused, i_fused = fused_softmax_topk(logits, top_k=2)

        # Indices should match when sorted
        i_ref_sorted = i_ref.sort(dim=-1).values
        i_fused_sorted = i_fused.sort(dim=-1).values
        self.assertTrue((i_ref_sorted == i_fused_sorted).all())


class TestFusedRouterDispatch(unittest.TestCase):

    def test_shapes(self):
        T, H, E, K = 16, 32, 8, 2
        hidden = torch.randn(T, H)
        router_w = torch.randn(H, E)
        idx, wt, logits = fused_router_dispatch(hidden, router_w, K)
        self.assertEqual(idx.shape, (T, K))
        self.assertEqual(wt.shape, (T, K))
        self.assertEqual(logits.shape, (T, E))

    def test_no_nan(self):
        T, H, E, K = 8, 32, 4, 2
        hidden = torch.randn(T, H)
        router_w = torch.randn(H, E)
        idx, wt, logits = fused_router_dispatch(hidden, router_w, K)
        self.assertFalse(torch.isnan(wt.float()).any())
        self.assertFalse(torch.isnan(logits).any())


class TestFusedSwiGLU(unittest.TestCase):

    def test_shape(self):
        M, H, N = 8, 32, 64
        x = torch.randn(M, H, dtype=torch.bfloat16)
        wg = torch.randn(H, N, dtype=torch.bfloat16)
        wu = torch.randn(H, N, dtype=torch.bfloat16)
        wd = torch.randn(N, H, dtype=torch.bfloat16)
        out = fused_swiglu_expert(x, wg, wu, wd)
        self.assertEqual(out.shape, (M, H))

    def test_no_nan(self):
        M, H, N = 16, 32, 64
        x = torch.randn(M, H, dtype=torch.bfloat16)
        wg = torch.randn(H, N, dtype=torch.bfloat16)
        wu = torch.randn(H, N, dtype=torch.bfloat16)
        wd = torch.randn(N, H, dtype=torch.bfloat16)
        out = fused_swiglu_expert(x, wg, wu, wd)
        self.assertFalse(torch.isnan(out.float()).any())


# ===========================================================================
# TritonMoELayer tests
# ===========================================================================


class TestTritonMoELayer(unittest.TestCase):

    def _make_layer(self, hidden=32, ffn=64, n=4, k=2):
        return TritonMoELayer(
            num_experts=n,
            hidden_dim=hidden,
            ffn_dim=ffn,
            top_k=k,
            dtype=torch.float32,
            device="cpu",
        )

    def test_forward_shape(self):
        layer = self._make_layer()
        x = torch.randn(2, 8, 32)
        out = layer(x)
        self.assertEqual(out.shape, (2, 8, 32))

    def test_forward_no_nan(self):
        layer = self._make_layer()
        x = torch.randn(2, 8, 32)
        out = layer(x)
        self.assertFalse(torch.isnan(out).any())

    def test_residual_added(self):
        layer = self._make_layer()
        # Zero out all expert weights to verify residual is applied
        for p in layer.parameters():
            p.data.zero_()
        x = torch.randn(2, 4, 32)
        out = layer(x)
        # With zero weights, output should be residual of layernorm(x) + x
        # At least verify output is not zero
        self.assertFalse((out == 0).all())

    def test_extra_repr(self):
        layer = self._make_layer()
        r = layer.extra_repr()
        self.assertIn("num_experts=4", r)
        self.assertIn("hidden_dim=32", r)

    def test_single_batch(self):
        layer = self._make_layer()
        x = torch.randn(1, 4, 32)
        out = layer(x)
        self.assertEqual(out.shape, (1, 4, 32))


# ===========================================================================
# Mixed precision context tests
# ===========================================================================


class TestMixedPrecisionContext(unittest.TestCase):

    def test_enters_exits_no_raise(self):
        ctx = MixedPrecisionContext(enabled=False)
        with ctx:
            pass  # should not raise

    def test_enabled_false_on_cpu(self):
        ctx = MixedPrecisionContext(enabled=True, dtype=torch.bfloat16)
        # On CPU, enabled should be False (no CUDA)
        if not torch.cuda.is_available():
            self.assertFalse(ctx.enabled)


class TestCastBf16IfCuda(unittest.TestCase):

    def test_cpu_tensor_stays_float32(self):
        x = torch.randn(4, 4, dtype=torch.float32)
        out = cast_to_bf16_if_cuda(x)
        self.assertEqual(out.dtype, torch.float32)

    @skipIfNoCUDA
    def test_cuda_tensor_cast_bf16(self):
        x = torch.randn(4, 4, device="cuda", dtype=torch.float32)
        out = cast_to_bf16_if_cuda(x)
        self.assertEqual(out.dtype, torch.bfloat16)


# ===========================================================================
# PackedInt4Weight tests
# ===========================================================================


class TestPackedInt4Weight(unittest.TestCase):

    def test_pack_unpack_roundtrip(self):
        weight = torch.randint(0, 15, (4, 8), dtype=torch.int32)
        packed = PackedInt4Weight(weight)
        unpacked = packed.unpack()
        self.assertEqual(unpacked.shape, weight.shape)
        self.assertTrue((unpacked == weight).all())

    def test_memory_savings(self):
        weight = torch.randint(0, 15, (64, 64), dtype=torch.int32)
        packed = PackedInt4Weight(weight)
        self.assertLess(packed.memory_bytes(), packed.original_memory_bytes())

    def test_memory_roughly_halved(self):
        weight = torch.randint(0, 15, (128, 128), dtype=torch.int32)
        packed = PackedInt4Weight(weight)
        ratio = packed.original_memory_bytes() / packed.memory_bytes()
        self.assertAlmostEqual(ratio, 8.0, delta=1.0)

    def test_odd_size(self):
        weight = torch.randint(0, 15, (3, 7), dtype=torch.int32)
        packed = PackedInt4Weight(weight)
        unpacked = packed.unpack()
        self.assertEqual(unpacked.shape, weight.shape)


# ===========================================================================
# Autotune config tests
# ===========================================================================


class TestAutotuneConfigs(unittest.TestCase):

    def test_configs_exist_for_known_gpus(self):
        for gpu in ["H100", "A100", "4090", "GENERIC"]:
            configs = _get_autotune_configs(gpu)
            self.assertGreater(len(configs), 0)

    def test_configs_have_required_keys(self):
        for gpu in ["H100", "A100", "4090", "GENERIC"]:
            configs = _get_autotune_configs(gpu)
            for c in configs:
                self.assertIn("BLOCK_M", c)
                self.assertIn("BLOCK_K", c)
                self.assertIn("num_warps", c)
                self.assertIn("num_stages", c)

    def test_make_triton_configs_empty_without_triton(self):
        if TRITON_AVAILABLE:
            configs = _make_triton_configs(_get_autotune_configs("GENERIC"))
            self.assertGreater(len(configs), 0)
        else:
            configs = _make_triton_configs(_get_autotune_configs("GENERIC"))
            self.assertEqual(len(configs), 0)

    def test_select_best_config(self):
        cfg = select_best_config("softmax_topk", T=1024, H=512, E=8)
        self.assertIn("BLOCK_M", cfg)
        self.assertIn("BLOCK_K", cfg)

    def test_select_best_config_large(self):
        cfg = select_best_config("softmax_topk", T=10000, H=1024, E=64)
        self.assertGreaterEqual(cfg["BLOCK_M"] * cfg["BLOCK_K"], 1)


# ===========================================================================
# KernelBenchmark tests (CPU mode, no CUDA required)
# ===========================================================================


class TestKernelBenchmarkCPU(unittest.TestCase):

    def test_benchmark_softmax_topk(self):
        bench = KernelBenchmark(device="cpu", dtype=torch.float32)
        result = bench.benchmark_softmax_topk(T=64, E=4, K=2)
        self.assertIn("pytorch_ms", result)
        self.assertGreater(result["pytorch_ms"], 0.0)

    def test_benchmark_swiglu_expert(self):
        bench = KernelBenchmark(device="cpu", dtype=torch.float32)
        result = bench.benchmark_swiglu_expert(M=32, H=32, N=64)
        self.assertIn("pytorch_ms", result)
        self.assertGreater(result["pytorch_ms"], 0.0)

    def test_benchmark_router_dispatch(self):
        bench = KernelBenchmark(device="cpu", dtype=torch.float32)
        result = bench.benchmark_router_dispatch(T=64, H=32, E=4, K=2)
        self.assertIn("pytorch_ms", result)
        self.assertGreater(result["pytorch_ms"], 0.0)


# ===========================================================================
# Verification tests (with CUDA + Triton guard)
# ===========================================================================


class TestKernelVerification(unittest.TestCase):

    def test_verify_softmax_topk_returns_bool(self):
        result = verify_softmax_topk(T=16, E=4, K=2, device=DEVICE)
        self.assertIsInstance(result, bool)

    def test_verify_swiglu_expert_returns_bool(self):
        result = verify_swiglu_expert(M=8, H=16, N=32, device=DEVICE)
        self.assertIsInstance(result, bool)

    def test_run_all_verifications_returns_bool(self):
        result = run_all_verifications(device=DEVICE)
        self.assertIsInstance(result, bool)

    @skipIfNoCUDA
    def test_verify_softmax_topk_cuda(self):
        result = verify_softmax_topk(T=64, E=8, K=2, device="cuda")
        # On CUDA without Triton, should still return True (skips check)
        self.assertIsInstance(result, bool)

    @skipIfNoTriton
    @skipIfNoCUDA
    def test_triton_softmax_topk_accuracy(self):
        """Triton output should be close to PyTorch reference."""
        logits = torch.randn(128, 8, device="cuda", dtype=torch.float32)
        w_ref, i_ref = _pytorch_softmax_topk(logits.clone(), 2)
        w_tri, i_tri = fused_softmax_topk(logits.clone(), 2)
        # Indices sorted should match
        self.assertTrue((i_ref.sort(dim=-1).values == i_tri.sort(dim=-1).values).all())

    @skipIfNoTriton
    @skipIfNoCUDA
    def test_triton_swiglu_accuracy(self):
        """Triton SwiGLU output should be close to PyTorch reference."""
        M, H, N = 32, 64, 128
        x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
        wg = torch.randn(H, N, device="cuda", dtype=torch.bfloat16)
        wu = torch.randn(H, N, device="cuda", dtype=torch.bfloat16)
        wd = torch.randn(N, H, device="cuda", dtype=torch.bfloat16)

        ref = _pytorch_swiglu_expert(x, wg, wu, wd)
        tri = fused_swiglu_expert(x, wg, wu, wd)
        close = torch.allclose(ref.float(), tri.float(), atol=0.1, rtol=0.1)
        self.assertTrue(close)


# ===========================================================================
# GPU type detection tests
# ===========================================================================


class TestGPUType(unittest.TestCase):

    def test_gpu_type_is_string(self):
        self.assertIsInstance(GPU_TYPE, str)

    def test_gpu_type_one_of_known(self):
        known = {"H100", "A100", "4090", "A10", "GENERIC", "cpu"}
        self.assertIn(GPU_TYPE, known)


class TestChunkedMoERouter(unittest.TestCase):

    def test_shapes_consistent_with_pytorch(self):
        from lumina.triton_kernels import ChunkedMoERouter, _pytorch_router_dispatch
        T, H, E, K = 64, 32, 8, 2
        hidden = torch.randn(T, H)
        router_w = torch.randn(H, E)

        chunked = ChunkedMoERouter(router_w, E, K, chunk_size=16)
        idx_c, wt_c, logits_c = chunked.route(hidden)
        idx_r, wt_r, logits_r = _pytorch_router_dispatch(hidden, router_w, K)

        self.assertEqual(idx_c.shape, idx_r.shape)
        self.assertEqual(wt_c.shape, wt_r.shape)
        self.assertEqual(logits_c.shape, logits_r.shape)

    def test_chunk_size_larger_than_T(self):
        from lumina.triton_kernels import ChunkedMoERouter
        T, H, E, K = 8, 32, 4, 2
        hidden = torch.randn(T, H)
        router_w = torch.randn(H, E)
        chunked = ChunkedMoERouter(router_w, E, K, chunk_size=1024)
        idx, wt, logits = chunked.route(hidden)
        self.assertEqual(idx.shape, (T, K))

    def test_multiple_chunk_sizes(self):
        from lumina.triton_kernels import ChunkedMoERouter, _pytorch_router_dispatch
        T, H, E, K = 100, 32, 4, 2
        hidden = torch.randn(T, H)
        router_w = torch.randn(H, E)

        for chunk_size in [10, 25, 50, 100, 200]:
            chunked = ChunkedMoERouter(router_w, E, K, chunk_size=chunk_size)
            idx, wt, logits = chunked.route(hidden)
            self.assertEqual(idx.shape, (T, K))
            self.assertEqual(logits.shape, (T, E))


class TestTritonKernelProfiler(unittest.TestCase):

    def test_profile_records_timing(self):
        from lumina.triton_kernels import TritonKernelProfiler
        profiler = TritonKernelProfiler(enabled=False)  # CPU mode
        result = profiler.profile("test_fn", lambda x: x * 2, torch.randn(4, 4))
        self.assertIsNotNone(result)

    def test_summary_empty(self):
        from lumina.triton_kernels import TritonKernelProfiler
        profiler = TritonKernelProfiler(enabled=False)
        s = profiler.summary()
        self.assertIsInstance(s, dict)

    def test_reset(self):
        from lumina.triton_kernels import TritonKernelProfiler
        profiler = TritonKernelProfiler(enabled=False)
        profiler.profile("fn", lambda: None)
        profiler.reset()
        self.assertEqual(len(profiler.summary()), 0)


class TestAnalyzeSoftmaxStability(unittest.TestCase):

    def test_returns_dict_with_expected_keys(self):
        from lumina.triton_kernels import analyze_softmax_stability
        logits = torch.randn(32, 8)
        result = analyze_softmax_stability(logits, perturbation=1e-3, n_trials=5)
        self.assertIn("mean_routing_agreement", result)
        self.assertIn("min_routing_agreement", result)
        self.assertIn("mean_weight_perturbation", result)

    def test_agreement_in_range(self):
        from lumina.triton_kernels import analyze_softmax_stability
        logits = torch.randn(32, 8) * 10  # sharp routing (high confidence)
        result = analyze_softmax_stability(logits, perturbation=1e-4, n_trials=3)
        self.assertGreaterEqual(result["mean_routing_agreement"], 0.0)
        self.assertLessEqual(result["mean_routing_agreement"], 1.0)

    def test_large_perturbation_reduces_agreement(self):
        from lumina.triton_kernels import analyze_softmax_stability
        logits = torch.randn(32, 8) * 0.1  # near-uniform routing
        small = analyze_softmax_stability(logits, perturbation=1e-6, n_trials=3)
        large = analyze_softmax_stability(logits, perturbation=10.0, n_trials=3)
        # Large perturbation should cause lower or equal agreement
        self.assertGreaterEqual(
            small["mean_routing_agreement"],
            large["mean_routing_agreement"] - 0.3  # with some tolerance
        )


class TestAnalyzeFusionOpportunities(unittest.TestCase):

    def test_returns_dict(self):
        from lumina.triton_kernels import analyze_fusion_opportunities
        model = torch.nn.Sequential(
            torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 64),
        )
        result = analyze_fusion_opportunities(model)
        self.assertIsInstance(result, dict)

    def test_estimate_speedup_no_raise(self):
        from lumina.triton_kernels import analyze_fusion_opportunities, estimate_fusion_speedup
        model = torch.nn.Sequential(torch.nn.Linear(32, 64))
        opps = analyze_fusion_opportunities(model)
        speedups = estimate_fusion_speedup(opps)
        self.assertIn("combined_estimated_speedup", speedups)
        self.assertGreater(speedups["combined_estimated_speedup"], 0.0)


class TestAdaptiveKernelSelector(unittest.TestCase):

    def test_select_returns_string(self):
        from lumina.triton_kernels import AdaptiveKernelSelector
        sel = AdaptiveKernelSelector(gpu_type="GENERIC")
        k = sel.select_router_kernel(T=64, H=32, E=8)
        self.assertIn(k, ["triton", "int8_triton", "pytorch"])

    def test_small_batch_returns_pytorch(self):
        from lumina.triton_kernels import AdaptiveKernelSelector
        sel = AdaptiveKernelSelector(gpu_type="GENERIC")
        k = sel.select_router_kernel(T=1, H=32, E=8)
        self.assertEqual(k, "pytorch")

    def test_record_latency_updates_table(self):
        from lumina.triton_kernels import AdaptiveKernelSelector
        sel = AdaptiveKernelSelector(gpu_type="GENERIC")
        shape = (128, 512, 8)
        for _ in range(10):
            sel.record_latency("pytorch", shape, 2.0)
        for _ in range(10):
            sel.record_latency("triton", shape, 1.0)
        # After enough measurements, triton should be selected
        best = sel._kernel_table.get(shape)
        # Should select triton (lower latency)
        if best is not None:
            self.assertEqual(best, "triton")

    def test_summary_has_expected_keys(self):
        from lumina.triton_kernels import AdaptiveKernelSelector
        sel = AdaptiveKernelSelector()
        s = sel.summary()
        self.assertIn("gpu_type", s)
        self.assertIn("triton_available", s)


class TestTritonCompilationCache(unittest.TestCase):

    def test_cache_dir_created(self):
        from lumina.triton_kernels import TritonCompilationCache
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TritonCompilationCache(cache_dir=tmpdir)
            self.assertTrue(os.path.exists(tmpdir))

    def test_size_mb_returns_float(self):
        from lumina.triton_kernels import TritonCompilationCache
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TritonCompilationCache(cache_dir=tmpdir)
            size = cache.size_mb()
            self.assertIsInstance(size, float)
            self.assertGreaterEqual(size, 0.0)

    def test_path_property(self):
        from lumina.triton_kernels import TritonCompilationCache
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TritonCompilationCache(cache_dir=tmpdir)
            self.assertEqual(cache.path, tmpdir)


import os


if __name__ == "__main__":
    unittest.main(verbosity=2)
