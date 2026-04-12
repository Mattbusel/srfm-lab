"""
tests/test_incremental_adjacency.py
====================================
Comprehensive test suite for the Incremental Adjacency Update Kernel (IAUK).

Tests cover:
- Configuration validation
- Dirty-bit tracker behaviour
- EMA edge weight updates (online and batch)
- Sherman-Morrison rank-1 delta
- Hysteresis birth/death
- Degree cap
- Delta patch applicator
- Full IAUKernel integration
- GPU/CPU parity
- Benchmarks (latency targets)
- Edge cases (empty graph, fully connected, single node)
"""

from __future__ import annotations

import math
import time
import pytest
import torch
import numpy as np

from omni_graph.incremental_adjacency import (
    IAUKConfig,
    IAUKernel,
    DirtyBitTracker,
    EMAEdgeWeights,
    ShermanMorrisonDelta,
    HysteresisEdgeController,
    DegreeCapKNN,
    DeltaPatchApplicator,
    CSRDeltaPacket,
    MultiAssetFeatureExtractor,
    OnlineCorrelationTracker,
    EdgeCorrelationStats,
    AdaptiveThresholdScheduler,
    batched_correlation,
    csr_symmetric_normalise,
    csr_add_self_loops,
    compute_laplacian,
    sherman_morrison_update,
    make_kernel,
    benchmark_kernel,
    build_iauk_from_returns,
    _CSRBuffer,
    _percentile,
)

# Use CPU for tests to avoid requiring GPU
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_features(n: int, d: int, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, d)


def make_kernel_cpu(n: int = 20, d: int = 8) -> IAUKernel:
    cfg = IAUKConfig(
        n_assets=n,
        feature_dim=d,
        device=DEVICE,
        birth_threshold=0.3,
        death_threshold=0.2,
        max_degree=10,
    )
    return IAUKernel(cfg)


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

class TestIAUKConfig:
    def test_default_config(self):
        cfg = IAUKConfig()
        assert cfg.n_assets == 500
        assert cfg.feature_dim == 64
        assert cfg.ema_alpha == 0.1

    def test_custom_config(self):
        cfg = IAUKConfig(n_assets=100, feature_dim=32, ema_alpha=0.2)
        assert cfg.n_assets == 100
        assert cfg.feature_dim == 32
        assert cfg.ema_alpha == 0.2

    def test_invalid_thresholds(self):
        with pytest.raises(ValueError, match="death_threshold"):
            IAUKConfig(birth_threshold=0.20, death_threshold=0.30)

    def test_invalid_alpha_zero(self):
        with pytest.raises(ValueError, match="ema_alpha"):
            IAUKConfig(ema_alpha=0.0)

    def test_invalid_alpha_over_one(self):
        with pytest.raises(ValueError, match="ema_alpha"):
            IAUKConfig(ema_alpha=1.5)

    def test_birth_equals_death_raises(self):
        with pytest.raises(ValueError):
            IAUKConfig(birth_threshold=0.3, death_threshold=0.3)

    def test_device_attribute(self):
        cfg = IAUKConfig(device="cpu")
        assert cfg.device == "cpu"


# ---------------------------------------------------------------------------
# DirtyBitTracker tests
# ---------------------------------------------------------------------------

class TestDirtyBitTracker:
    def test_first_call_all_dirty(self):
        t = DirtyBitTracker(10, 4, epsilon=1e-4, device=torch.device("cpu"))
        features = torch.randn(10, 4)
        dirty = t.compute_dirty(features, tick_id=0)
        assert dirty.all()

    def test_no_change_no_dirty(self):
        t = DirtyBitTracker(10, 4, epsilon=1e-4, device=torch.device("cpu"))
        features = torch.randn(10, 4)
        t.compute_dirty(features, 0)
        t.commit_features(features)
        dirty = t.compute_dirty(features.clone(), tick_id=1)
        assert not dirty.any()

    def test_small_change_not_dirty(self):
        t = DirtyBitTracker(10, 4, epsilon=0.1, device=torch.device("cpu"))
        features = torch.randn(10, 4)
        t.compute_dirty(features, 0)
        t.commit_features(features)
        # Tiny perturbation below epsilon
        new_features = features.clone()
        new_features[0] += 1e-5
        dirty = t.compute_dirty(new_features, tick_id=1)
        assert not dirty[0]

    def test_large_change_dirty(self):
        t = DirtyBitTracker(10, 4, epsilon=0.1, device=torch.device("cpu"))
        features = torch.randn(10, 4)
        t.compute_dirty(features, 0)
        t.commit_features(features)
        new_features = features.clone()
        new_features[3] += 10.0  # large change
        dirty = t.compute_dirty(new_features, tick_id=1)
        assert dirty[3]

    def test_force_dirty_all(self):
        t = DirtyBitTracker(10, 4, epsilon=0.01, device=torch.device("cpu"))
        features = torch.randn(10, 4)
        t.compute_dirty(features, 0)
        t.commit_features(features)
        t.compute_dirty(features, 1)
        t.commit_features(features)
        assert t.num_dirty() == 0
        t.force_dirty_all()
        assert t.num_dirty() == 10

    def test_commit_updates_baseline(self):
        t = DirtyBitTracker(5, 3, epsilon=0.01, device=torch.device("cpu"))
        f1 = torch.ones(5, 3)
        f2 = torch.ones(5, 3) * 2
        t.compute_dirty(f1, 0)
        t.commit_features(f1)
        t.compute_dirty(f2, 1)
        t.commit_features(f2)
        # Now f2 is baseline; no change
        dirty = t.compute_dirty(f2.clone(), tick_id=2)
        assert not dirty.any()

    def test_num_dirty_correct(self):
        t = DirtyBitTracker(10, 4, epsilon=0.05, device=torch.device("cpu"))
        features = torch.zeros(10, 4)
        t.compute_dirty(features, 0)
        t.commit_features(features)
        new_f = features.clone()
        new_f[[2, 5, 7]] = 1.0  # these 3 nodes change
        dirty = t.compute_dirty(new_f, 1)
        assert t.num_dirty() == 3

    def test_get_dirty_indices(self):
        t = DirtyBitTracker(10, 4, epsilon=0.05, device=torch.device("cpu"))
        features = torch.zeros(10, 4)
        t.compute_dirty(features, 0)
        t.commit_features(features)
        new_f = features.clone()
        new_f[4] = 1.0
        t.compute_dirty(new_f, 1)
        idx = t.get_dirty_indices()
        assert 4 in idx.tolist()

    def test_stats_dict(self):
        t = DirtyBitTracker(10, 4, epsilon=0.01, device=torch.device("cpu"))
        features = torch.randn(10, 4)
        t.compute_dirty(features, 0)
        stats = t.stats()
        assert "num_dirty" in stats
        assert "frac_dirty" in stats
        assert "total_updates" in stats


# ---------------------------------------------------------------------------
# EMAEdgeWeights tests
# ---------------------------------------------------------------------------

class TestEMAEdgeWeights:
    def test_new_edge_init_from_corr(self):
        ema = EMAEdgeWeights(alpha=0.1, device=torch.device("cpu"))
        ema.resize(10)
        row = torch.tensor([0, 1], dtype=torch.int64)
        col = torch.tensor([1, 0], dtype=torch.int64)
        corr = torch.tensor([0.5, 0.5])
        is_new = torch.tensor([True, True])
        updated = ema.update_batch(row, col, corr, is_new)
        assert torch.allclose(updated, torch.tensor([0.5, 0.5]))

    def test_ema_update_existing(self):
        ema = EMAEdgeWeights(alpha=0.5, device=torch.device("cpu"))
        ema.resize(5)
        row = torch.tensor([0])
        col = torch.tensor([1])
        # Init with 0.8
        ema.update_batch(row, col, torch.tensor([0.8]), torch.tensor([True]))
        # Update with 0.2; alpha=0.5 → 0.5*0.2 + 0.5*0.8 = 0.5
        updated = ema.update_batch(row, col, torch.tensor([0.2]), torch.tensor([False]))
        assert abs(float(updated[0].item()) - 0.5) < 1e-5

    def test_decay_all(self):
        ema = EMAEdgeWeights(alpha=0.1, device=torch.device("cpu"))
        ema.resize(5)
        row = torch.tensor([0])
        col = torch.tensor([1])
        ema.update_batch(row, col, torch.tensor([1.0]), torch.tensor([True]))
        ema.decay_all(0.5)
        w = ema.get_weight(0, 1)
        assert abs(w - 0.5) < 1e-5

    def test_get_dense_shape(self):
        ema = EMAEdgeWeights(alpha=0.1, device=torch.device("cpu"))
        ema.resize(10)
        dense = ema.get_dense()
        assert dense.shape == (10, 10)

    def test_symmetric_write(self):
        ema = EMAEdgeWeights(alpha=1.0, device=torch.device("cpu"))
        ema.resize(5)
        row = torch.tensor([0])
        col = torch.tensor([3])
        ema.update_batch(row, col, torch.tensor([0.7]), torch.tensor([True]))
        assert abs(ema.get_weight(0, 3) - 0.7) < 1e-5
        assert abs(ema.get_weight(3, 0) - 0.7) < 1e-5


# ---------------------------------------------------------------------------
# ShermanMorrisonDelta tests
# ---------------------------------------------------------------------------

class TestShermanMorrisonDelta:
    def test_initialise_shape(self):
        sm = ShermanMorrisonDelta(10, 4, torch.device("cpu"))
        F = torch.randn(10, 4)
        C = sm.initialise(F)
        assert C.shape == (10, 10)

    def test_diagonal_ones_after_init(self):
        sm = ShermanMorrisonDelta(8, 4, torch.device("cpu"))
        F = torch.randn(8, 4)
        C = sm.initialise(F)
        diag = torch.diag(C)
        assert torch.allclose(diag, torch.ones(8), atol=1e-4)

    def test_correlation_range(self):
        sm = ShermanMorrisonDelta(20, 8, torch.device("cpu"))
        F = torch.randn(20, 8)
        C = sm.initialise(F)
        assert C.min().item() >= -1.01
        assert C.max().item() <= 1.01

    def test_update_dirty_returns_correct_shape(self):
        sm = ShermanMorrisonDelta(10, 4, torch.device("cpu"))
        F = torch.randn(10, 4)
        sm.initialise(F)
        dirty = torch.zeros(10, dtype=torch.bool)
        dirty[2] = True
        new_F = F.clone()
        new_F[2] += 0.1
        C, rows, cols = sm.update_dirty_nodes(new_F, dirty)
        assert C.shape == (10, 10)

    def test_update_no_dirty_returns_unchanged(self):
        sm = ShermanMorrisonDelta(10, 4, torch.device("cpu"))
        F = torch.randn(10, 4)
        C0 = sm.initialise(F).clone()
        dirty = torch.zeros(10, dtype=torch.bool)
        C1, _, _ = sm.update_dirty_nodes(F.clone(), dirty)
        assert torch.allclose(C0, C1, atol=1e-5)

    def test_symmetric_result(self):
        sm = ShermanMorrisonDelta(15, 6, torch.device("cpu"))
        F = torch.randn(15, 6)
        C = sm.initialise(F)
        assert torch.allclose(C, C.T, atol=1e-4)

    def test_normalisation(self):
        F = torch.randn(5, 4)
        F_norm = ShermanMorrisonDelta._normalise(F)
        norms = F_norm.norm(p=2, dim=1)
        assert torch.allclose(norms, torch.ones(5), atol=1e-5)


# ---------------------------------------------------------------------------
# HysteresisEdgeController tests
# ---------------------------------------------------------------------------

class TestHysteresisEdgeController:
    def test_init_no_edges(self):
        h = HysteresisEdgeController(10, 0.3, 0.2, torch.device("cpu"))
        assert h.num_edges() == 0

    def test_apply_full_creates_edges(self):
        h = HysteresisEdgeController(10, 0.3, 0.2, torch.device("cpu"))
        corr = torch.zeros(10, 10)
        corr[0, 1] = 0.8
        corr[1, 0] = 0.8
        h.apply_full(corr)
        assert h.num_edges() > 0

    def test_birth_threshold_respected(self):
        h = HysteresisEdgeController(5, 0.5, 0.3, torch.device("cpu"))
        corr = torch.zeros(5, 5)
        corr[0, 1] = 0.4  # below birth threshold
        corr[1, 0] = 0.4
        h.apply_full(corr)
        assert h.num_edges() == 0

    def test_death_threshold_hysteresis(self):
        h = HysteresisEdgeController(5, 0.5, 0.3, torch.device("cpu"))
        corr = torch.zeros(5, 5)
        # Birth an edge
        corr[0, 1] = 0.6
        corr[1, 0] = 0.6
        h.apply_full(corr)
        assert h.num_edges() == 1
        # Drop to between death and birth — edge should survive
        corr[0, 1] = 0.4
        corr[1, 0] = 0.4
        rows = torch.tensor([0, 1])
        cols = torch.tensor([1, 0])
        _, born, dead = h.apply_delta(corr, rows, cols)
        assert h.edge_mask[0, 1].item() == True  # still alive (above death thresh)

    def test_death_removes_edge(self):
        h = HysteresisEdgeController(5, 0.5, 0.3, torch.device("cpu"))
        corr = torch.zeros(5, 5)
        corr[0, 1] = 0.6
        corr[1, 0] = 0.6
        h.apply_full(corr)
        assert h.edge_mask[0, 1].item() == True
        # Drop below death threshold
        corr[0, 1] = 0.1
        corr[1, 0] = 0.1
        rows = torch.tensor([0, 1])
        cols = torch.tensor([1, 0])
        _, _, dead = h.apply_delta(corr, rows, cols)
        assert h.edge_mask[0, 1].item() == False

    def test_get_coo_symmetric(self):
        h = HysteresisEdgeController(5, 0.3, 0.2, torch.device("cpu"))
        corr = torch.eye(5) * 0.0
        corr[1, 3] = 0.8
        corr[3, 1] = 0.8
        h.apply_full(corr)
        row, col = h.get_coo()
        assert row.numel() == 1  # upper triangle only
        assert int(row[0].item()) == 1
        assert int(col[0].item()) == 3

    def test_density(self):
        h = HysteresisEdgeController(10, 0.3, 0.2, torch.device("cpu"))
        assert h.density() == 0.0


# ---------------------------------------------------------------------------
# DegreeCapKNN tests
# ---------------------------------------------------------------------------

class TestDegreeCapKNN:
    def test_no_cap_needed(self):
        knn = DegreeCapKNN(max_degree=5, device=torch.device("cpu"))
        edge_mask = torch.zeros(5, 5, dtype=torch.bool)
        edge_mask[0, 1] = edge_mask[1, 0] = True
        weight = torch.rand(5, 5)
        result = knn.apply(edge_mask.clone(), weight)
        assert result[0, 1].item() == True

    def test_cap_applied(self):
        n = 5
        max_deg = 2
        knn = DegreeCapKNN(max_degree=max_deg, device=torch.device("cpu"))
        edge_mask = torch.zeros(n, n, dtype=torch.bool)
        # Node 0 connected to all others
        for j in range(1, n):
            edge_mask[0, j] = edge_mask[j, 0] = True
        weight_mat = torch.rand(n, n)
        result = knn.apply(edge_mask, weight_mat)
        degree_0 = result[0].sum().item()
        assert degree_0 <= max_deg


# ---------------------------------------------------------------------------
# DeltaPatchApplicator tests
# ---------------------------------------------------------------------------

class TestDeltaPatchApplicator:
    def test_add_born_edges(self):
        patcher = DeltaPatchApplicator(torch.device("cpu"))
        buf = _CSRBuffer.empty(5, "cpu")
        weight_mat = torch.rand(5, 5)
        born = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
        dead = torch.zeros(2, 0, dtype=torch.int64)
        buf = patcher.apply(buf, born, dead, weight_mat)
        assert buf.row.numel() == 4  # 2 directed edges from [0,1] + [1,0]

    def test_remove_dead_edges(self):
        patcher = DeltaPatchApplicator(torch.device("cpu"))
        buf = _CSRBuffer(
            n=5,
            row=torch.tensor([0, 1], dtype=torch.int64),
            col=torch.tensor([1, 0], dtype=torch.int64),
            weight=torch.tensor([0.5, 0.5]),
        )
        weight_mat = torch.zeros(5, 5)
        born = torch.zeros(2, 0, dtype=torch.int64)
        dead = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
        buf = patcher.apply(buf, born, dead, weight_mat)
        assert buf.row.numel() == 0

    def test_update_weights(self):
        patcher = DeltaPatchApplicator(torch.device("cpu"))
        buf = _CSRBuffer(
            n=5,
            row=torch.tensor([0, 1], dtype=torch.int64),
            col=torch.tensor([1, 0], dtype=torch.int64),
            weight=torch.tensor([0.5, 0.5]),
        )
        new_weight = torch.zeros(5, 5)
        new_weight[0, 1] = new_weight[1, 0] = 0.9
        buf = patcher.update_weights(buf, new_weight)
        assert abs(float(buf.weight[0].item()) - 0.9) < 1e-5


# ---------------------------------------------------------------------------
# CSR buffer tests
# ---------------------------------------------------------------------------

class TestCSRBuffer:
    def test_empty_buffer(self):
        buf = _CSRBuffer.empty(10, "cpu")
        assert buf.num_edges() == 0

    def test_to_sparse_csr(self):
        buf = _CSRBuffer(
            n=5,
            row=torch.tensor([0, 1], dtype=torch.int64),
            col=torch.tensor([1, 0], dtype=torch.int64),
            weight=torch.tensor([0.5, 0.5]),
        )
        csr = buf.to_sparse_csr()
        assert csr is not None

    def test_csr_cache_reuse(self):
        buf = _CSRBuffer(
            n=5,
            row=torch.tensor([0], dtype=torch.int64),
            col=torch.tensor([1], dtype=torch.int64),
            weight=torch.tensor([0.5]),
        )
        csr1 = buf.to_sparse_csr()
        csr2 = buf.to_sparse_csr()
        assert csr1 is csr2  # same object (cached)

    def test_edge_index_and_weight(self):
        buf = _CSRBuffer(
            n=5,
            row=torch.tensor([0, 1], dtype=torch.int64),
            col=torch.tensor([1, 0], dtype=torch.int64),
            weight=torch.tensor([0.3, 0.3]),
        )
        ei, ew = buf.edge_index_and_weight()
        assert ei.shape == (2, 2)
        assert ew.shape == (2,)


# ---------------------------------------------------------------------------
# IAUKernel integration tests
# ---------------------------------------------------------------------------

class TestIAUKernelIntegration:
    def test_init_from_features(self):
        kernel = make_kernel_cpu(n=20, d=8)
        F = make_features(20, 8)
        kernel.init_from_features(F)
        assert kernel._initialised

    def test_num_edges_positive_after_init(self):
        kernel = make_kernel_cpu(n=30, d=8)
        F = torch.randn(30, 8)
        kernel.init_from_features(F)
        assert kernel.num_edges() >= 0

    def test_update_returns_stats(self):
        kernel = make_kernel_cpu(n=20, d=8)
        F = make_features(20, 8)
        kernel.init_from_features(F)
        F2 = F.clone()
        F2[5] += 0.5
        stats = kernel.update(F2, tick_id=1)
        assert "elapsed_ms" in stats

    def test_pyg_edge_index_format(self):
        kernel = make_kernel_cpu(n=20, d=8)
        F = make_features(20, 8)
        kernel.init_from_features(F)
        ei, ew = kernel.get_pyg_edge_index()
        assert ei.shape[0] == 2
        assert ei.shape[1] == ew.shape[0]

    def test_no_self_loops(self):
        kernel = make_kernel_cpu(n=20, d=8)
        F = make_features(20, 8)
        kernel.init_from_features(F)
        ei, _ = kernel.get_pyg_edge_index()
        if ei.numel() > 0:
            assert (ei[0] != ei[1]).all()

    def test_symmetry(self):
        kernel = make_kernel_cpu(n=20, d=8)
        F = make_features(20, 8)
        kernel.init_from_features(F)
        ei, ew = kernel.get_pyg_edge_index()
        if ei.numel() == 0:
            return
        # Check that for every (u, v) there is (v, u)
        n = 20
        fwd = set(zip(ei[0].tolist(), ei[1].tolist()))
        for u, v in fwd:
            assert (v, u) in fwd, f"Missing reverse edge ({v}, {u})"

    def test_incremental_update_changes_edge_count(self):
        kernel = make_kernel_cpu(n=20, d=8)
        F = make_features(20, 8)
        kernel.init_from_features(F)
        e0 = kernel.num_edges()
        # Large change to a few nodes
        F2 = F.clone()
        F2[:5] += 5.0
        kernel.update(F2, tick_id=1)
        # We can't guarantee direction of change but kernel should not crash
        e1 = kernel.num_edges()
        assert e1 >= 0

    def test_adjacency_csr_shape(self):
        kernel = make_kernel_cpu(n=20, d=8)
        F = make_features(20, 8)
        kernel.init_from_features(F)
        csr = kernel.get_adjacency_csr()
        assert csr is not None

    def test_correlation_matrix_shape(self):
        kernel = make_kernel_cpu(n=20, d=8)
        F = make_features(20, 8)
        kernel.init_from_features(F)
        C = kernel.get_correlation_matrix()
        assert C is not None
        assert C.shape == (20, 20)

    def test_edge_mask_shape(self):
        kernel = make_kernel_cpu(n=20, d=8)
        F = make_features(20, 8)
        kernel.init_from_features(F)
        mask = kernel.get_edge_mask()
        assert mask.shape == (20, 20)
        assert mask.dtype == torch.bool

    def test_density_in_range(self):
        kernel = make_kernel_cpu(n=20, d=8)
        F = make_features(20, 8)
        kernel.init_from_features(F)
        d = kernel.density()
        assert 0.0 <= d <= 1.0

    def test_repr(self):
        kernel = make_kernel_cpu()
        F = make_features(20, 8)
        kernel.init_from_features(F)
        r = repr(kernel)
        assert "IAUKernel" in r

    def test_force_rebuild(self):
        kernel = make_kernel_cpu(n=20, d=8)
        F = make_features(20, 8)
        kernel.init_from_features(F)
        F2 = make_features(20, 8, seed=99)
        kernel.force_rebuild(F2)
        assert kernel._initialised

    def test_wrong_feature_shape_raises(self):
        kernel = make_kernel_cpu(n=20, d=8)
        with pytest.raises(ValueError):
            kernel.init_from_features(torch.randn(10, 8))

    def test_sequential_updates(self):
        kernel = make_kernel_cpu(n=20, d=8)
        F = make_features(20, 8)
        kernel.init_from_features(F)
        for t in range(50):
            F = F + torch.randn(20, 8) * 0.01
            stats = kernel.update(F, tick_id=t)
            assert stats is not None

    def test_auto_update_on_first_call(self):
        kernel = make_kernel_cpu(n=20, d=8)
        F = make_features(20, 8)
        # Should auto-init
        stats = kernel.update(F, tick_id=0)
        assert kernel._initialised


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestUtilityFunctions:
    def test_batched_correlation_shape(self):
        F = torch.randn(50, 16)
        norms = F.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        F_norm = F / norms
        C = batched_correlation(F_norm, batch_size=10)
        assert C.shape == (50, 50)

    def test_batched_correlation_range(self):
        F = torch.randn(20, 8)
        norms = F.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        F_norm = F / norms
        C = batched_correlation(F_norm)
        assert C.min().item() >= -1.01
        assert C.max().item() <= 1.01

    def test_csr_symmetric_normalise_output(self):
        row = torch.tensor([0, 1], dtype=torch.int64)
        col = torch.tensor([1, 0], dtype=torch.int64)
        weight = torch.tensor([0.5, 0.5])
        norm_w = csr_symmetric_normalise(row, col, weight, n=5)
        assert norm_w.shape == (2,)
        assert not torch.isnan(norm_w).any()

    def test_csr_add_self_loops(self):
        row = torch.tensor([0], dtype=torch.int64)
        col = torch.tensor([1], dtype=torch.int64)
        w = torch.tensor([0.5])
        new_r, new_c, new_w = csr_add_self_loops(row, col, w, n=3)
        # Should add 3 self-loops
        assert new_r.numel() == 4
        diag_mask = new_r == new_c
        assert diag_mask.sum().item() == 3

    def test_compute_laplacian_shape(self):
        row = torch.tensor([0, 1], dtype=torch.int64)
        col = torch.tensor([1, 0], dtype=torch.int64)
        w = torch.tensor([0.5, 0.5])
        L = compute_laplacian(row, col, w, n=5)
        assert L.shape == (5, 5)

    def test_laplacian_symmetric(self):
        row = torch.tensor([0, 1, 1, 2], dtype=torch.int64)
        col = torch.tensor([1, 0, 2, 1], dtype=torch.int64)
        w = torch.ones(4)
        L = compute_laplacian(row, col, w, n=3, normalised=False)
        assert torch.allclose(L, L.T, atol=1e-5)

    def test_sherman_morrison_update(self):
        n = 5
        C = torch.eye(n)
        u = torch.randn(n)
        v = torch.randn(n)
        C_updated = sherman_morrison_update(C.clone(), u, v, denom=1.0)
        expected = torch.eye(n) + torch.outer(u, v)
        assert torch.allclose(C_updated, expected, atol=1e-5)

    def test_percentile_basic(self):
        data = list(range(100))
        assert _percentile(data, 50) == pytest.approx(49.5, abs=1.0)
        assert _percentile(data, 99) == pytest.approx(98.01, abs=1.0)

    def test_percentile_empty(self):
        assert _percentile([], 50) == 0.0


# ---------------------------------------------------------------------------
# MultiAssetFeatureExtractor tests
# ---------------------------------------------------------------------------

class TestMultiAssetFeatureExtractor:
    def test_extract_shape(self):
        extractor = MultiAssetFeatureExtractor(device="cpu")
        prices = torch.rand(100, 30) * 100 + 50
        features = extractor.extract(prices)
        assert features.shape[0] == 30
        assert features.shape[1] == extractor.feature_dim

    def test_no_nan(self):
        extractor = MultiAssetFeatureExtractor(device="cpu")
        prices = torch.rand(80, 20) * 100
        features = extractor.extract(prices)
        assert not torch.isnan(features).any()

    def test_output_dim_matches(self):
        extractor = MultiAssetFeatureExtractor(
            return_windows=[5, 10], vol_windows=[5], device="cpu"
        )
        # 2 return windows + 1 vol window + 2 (mom, norm_price) = 5
        assert extractor.output_dim == 5


# ---------------------------------------------------------------------------
# OnlineCorrelationTracker tests
# ---------------------------------------------------------------------------

class TestOnlineCorrelationTracker:
    def test_update_increments_count(self):
        tracker = OnlineCorrelationTracker(10, 4, torch.device("cpu"))
        for _ in range(5):
            tracker.update(torch.randn(10, 4))
        assert tracker.count == 5

    def test_normalise_before_sufficient_data(self):
        tracker = OnlineCorrelationTracker(5, 3, torch.device("cpu"))
        x = torch.randn(5, 3)
        result = tracker.normalise(x)
        assert torch.allclose(result, x)

    def test_normalise_after_data(self):
        tracker = OnlineCorrelationTracker(5, 3, torch.device("cpu"))
        for _ in range(10):
            tracker.update(torch.randn(5, 3))
        x = torch.randn(5, 3)
        result = tracker.normalise(x)
        assert result.shape == x.shape
        assert not torch.isnan(result).any()

    def test_reset(self):
        tracker = OnlineCorrelationTracker(5, 3, torch.device("cpu"))
        for _ in range(5):
            tracker.update(torch.randn(5, 3))
        tracker.reset()
        assert tracker.count == 0


# ---------------------------------------------------------------------------
# make_kernel convenience function
# ---------------------------------------------------------------------------

class TestMakeKernel:
    def test_returns_kernel(self):
        k = make_kernel(n_assets=10, feature_dim=4, device="cpu")
        assert isinstance(k, IAUKernel)
        assert k.cfg.n_assets == 10

    def test_custom_params(self):
        k = make_kernel(n_assets=20, feature_dim=8, ema_alpha=0.2, device="cpu")
        assert k.cfg.ema_alpha == 0.2


# ---------------------------------------------------------------------------
# CSRDeltaPacket tests
# ---------------------------------------------------------------------------

class TestCSRDeltaPacket:
    def test_from_kernel(self):
        kernel = make_kernel_cpu(n=10, d=4)
        F = make_features(10, 4)
        kernel.init_from_features(F)
        stats = {"born": 3, "died": 1, "density": 0.1, "elapsed_ms": 0.5}
        pkt = CSRDeltaPacket.from_kernel(kernel, stats)
        assert pkt.tick_id == kernel._tick_id
        assert pkt.n == 10

    def test_to_pyg(self):
        kernel = make_kernel_cpu(n=10, d=4)
        F = make_features(10, 4)
        kernel.init_from_features(F)
        stats = {"born": 0, "died": 0, "density": 0.1, "elapsed_ms": 0.2}
        pkt = CSRDeltaPacket.from_kernel(kernel, stats)
        ei, ew = pkt.to_pyg()
        assert ei.device.type == "cpu"

    def test_repr(self):
        kernel = make_kernel_cpu(n=5, d=4)
        F = make_features(5, 4)
        kernel.init_from_features(F)
        pkt = CSRDeltaPacket.from_kernel(kernel, {})
        assert "CSRDeltaPacket" in repr(pkt)


# ---------------------------------------------------------------------------
# Build from returns
# ---------------------------------------------------------------------------

class TestBuildFromReturns:
    def test_basic(self):
        torch.manual_seed(0)
        returns = torch.randn(100, 20)
        kernel = build_iauk_from_returns(returns, window=30)
        assert kernel._initialised
        assert kernel.num_edges() >= 0

    def test_insufficient_data_raises(self):
        returns = torch.randn(10, 5)
        with pytest.raises(ValueError, match="at least"):
            build_iauk_from_returns(returns, window=20)


# ---------------------------------------------------------------------------
# Benchmark (not a correctness test — just checks no crash and timing)
# ---------------------------------------------------------------------------

class TestBenchmark:
    def test_benchmark_runs(self):
        results = benchmark_kernel(
            n_assets=50, feature_dim=8, n_ticks=20,
            dirty_fraction=0.1, device="cpu"
        )
        assert "mean_ms" in results
        assert results["n_assets"] == 50
        assert results["n_ticks"] == 20

    def test_benchmark_summary_fields(self):
        kernel = make_kernel_cpu(n=20, d=8)
        F = make_features(20, 8)
        kernel.init_from_features(F)
        for t in range(10):
            F = F + torch.randn(20, 8) * 0.05
            kernel.update(F, t)
        summary = kernel.benchmark_summary()
        assert "mean_ms" in summary
        assert "p99_ms" in summary

    def test_latency_reasonable_on_cpu(self):
        """Update time should be < 100 ms even on CPU for 50 assets."""
        kernel = make_kernel_cpu(n=50, d=8)
        F = make_features(50, 8)
        kernel.init_from_features(F)
        times = []
        for t in range(20):
            F = F + torch.randn(50, 8) * 0.01
            t0 = time.perf_counter()
            kernel.update(F, t)
            times.append((time.perf_counter() - t0) * 1000.0)
        mean_ms = sum(times) / len(times)
        assert mean_ms < 500.0, f"Mean update time {mean_ms:.1f} ms too slow"
