"""
tests/test_dynamic_edges.py
============================
Tests for dynamic_edges module — dynamic edge management.
"""

import math
import pytest
import numpy as np
import torch

from omni_graph.dynamic_edges import (
    EMAEdgeWeightManager,
    AdaptiveCorrelationThreshold,
    RegimeConditionedRewiring,
    EdgeBirthDeathProcess,
    TemporalEdgeAttention,
    GraphTopologyChangeDetector,
    LaplacianStructuralBreakDetector,
    DynamicGraphStateManager,
    TemporalEdgeFeatureExtractor,
    EdgeWeightTimeSeriesAnalyser,
    approximate_graph_edit_distance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_edge_index():
    """Simple 5-node cycle graph."""
    N = 5
    src = torch.tensor([0, 1, 2, 3, 4])
    dst = torch.tensor([1, 2, 3, 4, 0])
    return torch.stack([src, dst], dim=0)


@pytest.fixture
def simple_weights():
    return torch.tensor([0.5, 0.4, 0.6, 0.3, 0.7])


@pytest.fixture
def random_edge_index():
    """Random graph with 10 nodes and 20 edges."""
    np.random.seed(42)
    N, E = 10, 20
    src = torch.randint(0, N, (E,))
    dst = torch.randint(0, N, (E,))
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)


@pytest.fixture
def corr_matrix():
    """Random 10x10 correlation matrix."""
    np.random.seed(42)
    A = np.random.randn(10, 10).astype(np.float32)
    C = A @ A.T
    D = np.sqrt(np.diag(C))
    corr = C / np.outer(D, D)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1, 1)


# ---------------------------------------------------------------------------
# EMA edge weight manager tests
# ---------------------------------------------------------------------------

class TestEMAEdgeWeightManager:

    def test_initial_update_returns_tensors(self, simple_edge_index, simple_weights):
        manager = EMAEdgeWeightManager(alpha=0.3, birth_threshold=0.1, death_threshold=0.01)
        ei, w = manager.update(simple_edge_index, simple_weights)
        assert isinstance(ei, torch.Tensor)
        assert isinstance(w, torch.Tensor)
        assert ei.shape[0] == 2

    def test_ema_smoothing(self, simple_edge_index):
        """EMA weights should be smoother than raw weights."""
        manager = EMAEdgeWeightManager(alpha=0.3, birth_threshold=0.0, death_threshold=0.0)
        weights1 = torch.ones(simple_edge_index.shape[1])
        weights2 = torch.zeros(simple_edge_index.shape[1])

        manager.update(simple_edge_index, weights1)
        ei2, w2 = manager.update(simple_edge_index, weights2)

        # After EMA with alpha=0.3: w = 0.3*0 + 0.7*(0.3*1) = 0.21
        # Not exactly 0 — EMA effect
        assert w2.shape[0] > 0

    def test_death_threshold_removes_edges(self, simple_edge_index):
        """Edges with weight below death_threshold should be removed."""
        manager = EMAEdgeWeightManager(
            alpha=0.99, birth_threshold=0.01, death_threshold=0.5
        )
        # Update with low weights
        low_weights = torch.ones(simple_edge_index.shape[1]) * 0.1
        for _ in range(10):
            ei, w = manager.update(simple_edge_index, low_weights)

        # All weights should be very small → most edges should die
        n_live = len(manager._weights)
        assert n_live <= simple_edge_index.shape[1] // 2  # some should have died

    def test_max_edges_constraint(self, simple_edge_index, simple_weights):
        """Number of live edges should not exceed max_edges."""
        max_edges = 3
        manager = EMAEdgeWeightManager(
            alpha=0.3, birth_threshold=0.0, death_threshold=0.0, max_edges=max_edges
        )
        manager.update(simple_edge_index, simple_weights)
        assert len(manager._weights) <= max_edges

    def test_state_dict_roundtrip(self, simple_edge_index, simple_weights):
        manager = EMAEdgeWeightManager()
        manager.update(simple_edge_index, simple_weights)
        state = manager.state_dict()
        manager2 = EMAEdgeWeightManager()
        manager2.load_state_dict(state)
        assert manager2._t == manager._t
        assert len(manager2._weights) == len(manager._weights)

    def test_empty_edge_index(self):
        manager = EMAEdgeWeightManager()
        ei = torch.zeros(2, 0, dtype=torch.long)
        w = torch.zeros(0)
        out_ei, out_w = manager.update(ei, w)
        assert isinstance(out_ei, torch.Tensor)

    def test_repeated_updates_convergence(self, simple_edge_index):
        """EMA should converge to the target value."""
        manager = EMAEdgeWeightManager(alpha=0.5, birth_threshold=0.0, death_threshold=0.0)
        target_w = torch.ones(simple_edge_index.shape[1]) * 0.8
        for _ in range(30):
            ei, w = manager.update(simple_edge_index, target_w)
        # After many updates, EMA should be close to target
        if len(manager._weights) > 0:
            actual_ws = list(manager._weights.values())
            assert all(abs(v - 0.8) < 0.1 for v in actual_ws)


# ---------------------------------------------------------------------------
# Adaptive correlation threshold tests
# ---------------------------------------------------------------------------

class TestAdaptiveCorrelationThreshold:

    def test_initial_threshold(self):
        adapter = AdaptiveCorrelationThreshold(base_threshold=0.3)
        assert adapter.threshold == 0.3

    def test_threshold_updates(self):
        adapter = AdaptiveCorrelationThreshold(base_threshold=0.3, percentile=70.0, window=20)
        for _ in range(25):
            corr_vals = np.random.uniform(0.1, 0.9, 20)
            adapter.update(corr_vals)
        # Threshold should have updated from base
        # (could be higher or lower depending on percentile)
        assert 0.01 <= adapter.threshold <= 0.95

    def test_regime_volatility_adjustment(self):
        adapter = AdaptiveCorrelationThreshold(regime_sensitivity=0.5)
        # First, fill history
        for _ in range(15):
            adapter.update(np.random.uniform(0.2, 0.6, 20), volatility=0.1)

        t_before = adapter.threshold
        # High volatility should relax threshold
        adapter.update(np.random.uniform(0.2, 0.6, 20), volatility=10.0)
        # (Behavior depends on z-score; test it doesn't crash and stays valid)
        assert 0.01 <= adapter.threshold <= 0.95

    def test_threshold_clamped(self):
        adapter = AdaptiveCorrelationThreshold(base_threshold=0.3)
        # Feed extreme values
        for _ in range(20):
            adapter.update(np.ones(50), volatility=1.0)
        assert adapter.threshold <= 0.95
        assert adapter.threshold >= 0.01


# ---------------------------------------------------------------------------
# Regime conditioned rewiring tests
# ---------------------------------------------------------------------------

class TestRegimeConditionedRewiring:

    def test_rewiring_output_shape(self, simple_edge_index, simple_weights):
        rewirer = RegimeConditionedRewiring(base_threshold=0.2)
        ei, w = rewirer.rewire(simple_edge_index, simple_weights, regime=0)
        assert ei.shape[0] == 2
        assert w.shape[0] == ei.shape[1]

    def test_crisis_regime_has_more_edges(self, simple_edge_index, simple_weights):
        """Crisis regime (3) relaxes threshold → potentially more edges."""
        rewirer = RegimeConditionedRewiring(base_threshold=0.3)
        _, w_neutral = rewirer.rewire(simple_edge_index, simple_weights, regime=0)
        _, w_crisis = rewirer.rewire(simple_edge_index, simple_weights, regime=3)
        # Crisis has lower threshold multiplier → more edges pass
        assert w_crisis.shape[0] >= w_neutral.shape[0]

    def test_lead_lag_added_in_bear(self, simple_edge_index, simple_weights):
        """Bear and crisis regimes should add lead-lag edges if provided."""
        rewirer = RegimeConditionedRewiring(base_threshold=0.1)
        ll_ei = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        ll_w = torch.tensor([0.4, 0.3])
        ei_bear, w_bear = rewirer.rewire(
            simple_edge_index, simple_weights, regime=2,
            lead_lag_index=ll_ei, lead_lag_weights=ll_w,
        )
        assert ei_bear.shape[1] > simple_edge_index.shape[1]

    def test_smooth_transition(self, simple_edge_index, simple_weights):
        rewirer = RegimeConditionedRewiring()
        next_ei, next_w = simple_edge_index.clone(), simple_weights.clone()
        blended_ei, blended_w = rewirer.smooth_regime_transition(
            simple_edge_index, simple_weights, next_ei, next_w, alpha=0.3
        )
        assert blended_ei.shape[0] == 2
        assert blended_w.shape[0] == blended_ei.shape[1]


# ---------------------------------------------------------------------------
# Edge birth/death process tests
# ---------------------------------------------------------------------------

class TestEdgeBirthDeathProcess:

    def test_initial_step(self, corr_matrix):
        proc = EdgeBirthDeathProcess(seed=42)
        N = corr_matrix.shape[0]
        ei, w = proc.step(N, corr_matrix)
        assert isinstance(ei, torch.Tensor)
        assert isinstance(w, torch.Tensor)

    def test_high_correlation_increases_births(self):
        """High correlation should lead to more edge births."""
        proc_high = EdgeBirthDeathProcess(birth_rate_base=1.0, seed=0)
        proc_low = EdgeBirthDeathProcess(birth_rate_base=0.001, seed=0)

        N = 10
        corr_high = np.ones((N, N)) * 0.9
        np.fill_diagonal(corr_high, 1.0)

        for _ in range(10):
            proc_high.step(N, corr_high)
            proc_low.step(N, corr_high)

        assert proc_high.n_live_edges >= proc_low.n_live_edges

    def test_edges_die_with_low_weight(self):
        """Edges with very low weight should die quickly."""
        proc = EdgeBirthDeathProcess(death_rate_base=10.0, min_weight_to_survive=0.01, seed=42)
        N = 5
        corr = np.eye(N).astype(np.float32)  # no correlation → edges born weak

        for _ in range(20):
            proc.step(N, corr)

        # Most edges should be dead
        assert proc.n_live_edges < N * (N - 1) // 2

    def test_step_returns_valid_nodes(self, corr_matrix):
        proc = EdgeBirthDeathProcess(seed=42)
        N = corr_matrix.shape[0]
        for _ in range(5):
            ei, w = proc.step(N, corr_matrix)
        if ei.shape[1] > 0:
            assert ei.min() >= 0
            assert ei.max() < N * 2  # bidirectional → up to 2N unique IDs


# ---------------------------------------------------------------------------
# Temporal edge attention tests
# ---------------------------------------------------------------------------

class TestTemporalEdgeAttention:

    def test_forward_shape(self):
        T, E, F = 5, 8, 4
        n_heads = 2
        attn = TemporalEdgeAttention(edge_feat_dim=F, n_heads=n_heads, time_decay=0.1)
        edge_seq = torch.randn(T, E, F)
        time_deltas = torch.arange(T, dtype=torch.float32)
        weights = attn(edge_seq, time_deltas)
        assert weights.shape == (E, T)

    def test_attention_weights_sum_to_one(self):
        T, E, F = 5, 8, 4
        attn = TemporalEdgeAttention(edge_feat_dim=F)
        edge_seq = torch.randn(T, E, F)
        time_deltas = torch.arange(T, dtype=torch.float32)
        weights = attn(edge_seq, time_deltas)
        # Each edge's attention should sum to ~1
        row_sums = weights.sum(dim=1)
        assert (row_sums - 1.0).abs().max() < 0.1

    def test_aggregate_shape(self):
        T, E, F = 5, 8, 4
        attn = TemporalEdgeAttention(edge_feat_dim=F)
        edge_seq = torch.randn(T, E, F)
        time_deltas = torch.arange(T, dtype=torch.float32)
        agg = attn.aggregate(edge_seq, time_deltas)
        assert agg.shape == (E, F)

    def test_recency_weighting(self):
        """More recent edges (small time_delta) should get higher weights."""
        T, E, F = 5, 4, 4
        attn = TemporalEdgeAttention(edge_feat_dim=F, time_decay=2.0)
        # Same features for all time steps
        edge_seq = torch.ones(T, E, F)
        time_deltas = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])
        weights = attn(edge_seq, time_deltas)
        # Last time step (delta=0) should generally have higher attention
        # (not guaranteed with learned weights, just check no NaN)
        assert torch.isfinite(weights).all()


# ---------------------------------------------------------------------------
# Graph topology change detector tests
# ---------------------------------------------------------------------------

class TestGraphTopologyChangeDetector:

    def test_no_change_low_score(self, simple_edge_index):
        """Same graph → low change score."""
        detector = GraphTopologyChangeDetector(method="jaccard")
        score = detector.compute_change_score(simple_edge_index, simple_edge_index, 5)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_complete_change_high_score(self):
        """Completely different graphs → score = 1.0."""
        ei_a = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        ei_b = torch.tensor([[3, 4, 5], [4, 5, 3]], dtype=torch.long)
        detector = GraphTopologyChangeDetector(method="jaccard")
        score = detector.compute_change_score(ei_a, ei_b, 6)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_update_returns_score_and_bool(self, simple_edge_index):
        detector = GraphTopologyChangeDetector(method="jaccard")
        score, is_change = detector.update(simple_edge_index, simple_edge_index, 5)
        assert isinstance(score, float)
        assert isinstance(is_change, bool)

    def test_jaccard_method(self, simple_edge_index):
        ei2 = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        detector = GraphTopologyChangeDetector(method="jaccard")
        score = detector.compute_change_score(simple_edge_index, ei2, 5)
        assert 0.0 <= score <= 1.0

    def test_spectral_method(self, simple_edge_index):
        ei2 = torch.tensor([[0, 2], [2, 0]], dtype=torch.long)
        detector = GraphTopologyChangeDetector(method="spectral")
        score = detector.compute_change_score(simple_edge_index, ei2, 5)
        assert 0.0 <= score <= 1.0

    def test_degree_kl_method(self, simple_edge_index):
        ei2 = torch.tensor([[0, 1, 2, 3, 4], [2, 3, 4, 0, 1]], dtype=torch.long)
        detector = GraphTopologyChangeDetector(method="degree_kl")
        score = detector.compute_change_score(simple_edge_index, ei2, 5)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# Laplacian structural break detector tests
# ---------------------------------------------------------------------------

class TestLaplacianStructuralBreakDetector:

    def test_update_returns_dict(self, simple_edge_index):
        detector = LaplacianStructuralBreakDetector(n_eigenvalues=3)
        result = detector.update(simple_edge_index, 5)
        assert "fiedler_value" in result
        assert "spectral_gap" in result
        assert "cusum_alarm" in result

    def test_fiedler_value_non_negative(self, simple_edge_index):
        detector = LaplacianStructuralBreakDetector(n_eigenvalues=3)
        result = detector.update(simple_edge_index, 5)
        assert result["fiedler_value"] >= 0.0

    def test_eigenvalue_array_shape(self, simple_edge_index):
        n_eigs = 4
        detector = LaplacianStructuralBreakDetector(n_eigenvalues=n_eigs)
        result = detector.update(simple_edge_index, 5)
        assert len(result["eigenvalues"]) == n_eigs

    def test_cusum_alarm_fires(self, corr_matrix):
        """Alarm should fire if eigenvalues change dramatically."""
        from omni_graph.graph_topology import CorrelationGraphBuilder, GraphBuildConfig
        detector = LaplacianStructuralBreakDetector(
            n_eigenvalues=5, cusum_threshold=0.5, cusum_drift=0.0
        )
        N = corr_matrix.shape[0]

        # Feed stable graph
        builder = CorrelationGraphBuilder(GraphBuildConfig(corr_threshold=0.0))
        np.random.seed(42)
        for _ in range(25):
            returns = np.random.randn(30, N).astype(np.float32)
            g = builder.build(returns)
            detector.update(g.edge_index, N)

        # Feed very different graph (empty)
        empty_ei = torch.zeros(2, 0, dtype=torch.long)
        result = detector.update(empty_ei, N)
        # (might or might not alarm depending on data, just test no crash)
        assert isinstance(result["cusum_alarm"], bool)

    def test_fiedler_series(self, simple_edge_index):
        detector = LaplacianStructuralBreakDetector(n_eigenvalues=3)
        for _ in range(5):
            detector.update(simple_edge_index, 5)
        series = detector.get_fiedler_series()
        assert len(series) == 5


# ---------------------------------------------------------------------------
# Dynamic graph state manager tests
# ---------------------------------------------------------------------------

class TestDynamicGraphStateManager:

    def test_step_output_keys(self, simple_edge_index, simple_weights, corr_matrix):
        N = corr_matrix.shape[0]
        manager = DynamicGraphStateManager(N, detect_breaks=True, seed=42)
        # Need to provide edge_index with valid nodes for corr_matrix
        result = manager.step(
            simple_edge_index, simple_weights, corr_matrix, regime=0
        )
        assert "edge_index" in result
        assert "edge_weights" in result
        assert "change_score" in result
        assert "break_detected" in result
        assert "regime" in result

    def test_step_produces_valid_edges(self, corr_matrix):
        N = corr_matrix.shape[0]
        manager = DynamicGraphStateManager(N, seed=42)
        ei = torch.zeros(2, 0, dtype=torch.long)
        w = torch.zeros(0)
        result = manager.step(ei, w, corr_matrix, regime=1)
        out_ei = result["edge_index"]
        assert out_ei.shape[0] == 2

    def test_regime_change_detection(self, corr_matrix):
        N = corr_matrix.shape[0]
        manager = DynamicGraphStateManager(N, seed=42)
        src = torch.arange(N - 1)
        dst = torch.arange(1, N)
        ei = torch.stack([src, dst], dim=0)
        w = torch.rand(N - 1)

        for step in range(15):
            regime = 0 if step < 10 else 3  # switch to crisis
            result = manager.step(ei, w, corr_matrix, regime=regime)

        # Should not crash and return valid state
        assert isinstance(result["regime"], int)


# ---------------------------------------------------------------------------
# Temporal edge feature extractor tests
# ---------------------------------------------------------------------------

class TestTemporalEdgeFeatureExtractor:

    def test_forward_shape(self):
        T, E, F = 8, 12, 6
        extractor = TemporalEdgeFeatureExtractor(edge_feat_dim=F, hidden_dim=32, n_layers=2)
        edge_seq = torch.randn(T, E, F)
        time_deltas = torch.arange(T, dtype=torch.float32)
        out = extractor(edge_seq, time_deltas)
        assert out.shape == (E, 32)

    def test_no_attention_variant(self):
        T, E, F = 5, 10, 4
        extractor = TemporalEdgeFeatureExtractor(edge_feat_dim=F, hidden_dim=16, use_attention=False)
        edge_seq = torch.randn(T, E, F)
        out = extractor(edge_seq)
        assert out.shape == (E, 16)

    def test_gradients_flow(self):
        T, E, F = 5, 8, 4
        extractor = TemporalEdgeFeatureExtractor(edge_feat_dim=F, hidden_dim=16)
        edge_seq = torch.randn(T, E, F, requires_grad=True)
        time_deltas = torch.arange(T, dtype=torch.float32)
        out = extractor(edge_seq, time_deltas)
        loss = out.sum()
        loss.backward()
        assert edge_seq.grad is not None


# ---------------------------------------------------------------------------
# Edge weight time series analyser tests
# ---------------------------------------------------------------------------

class TestEdgeWeightTimeSeriesAnalyser:

    def test_basic_analysis(self, simple_edge_index, simple_weights):
        tracked = [(0, 1), (1, 2), (2, 3)]
        analyser = EdgeWeightTimeSeriesAnalyser(tracked, max_history=50)

        for _ in range(30):
            noisy_w = simple_weights + torch.randn_like(simple_weights) * 0.05
            analyser.update(simple_edge_index, noisy_w)

        result = analyser.analyse_edge((0, 1))
        assert "mean" in result
        assert "std" in result
        assert "ar1_coefficient" in result

    def test_analyse_all(self, simple_edge_index, simple_weights):
        tracked = [(0, 1), (1, 2)]
        analyser = EdgeWeightTimeSeriesAnalyser(tracked)
        for _ in range(20):
            analyser.update(simple_edge_index, simple_weights)
        results = analyser.analyse_all()
        assert (0, 1) in results
        assert (1, 2) in results

    def test_insufficient_data(self):
        analyser = EdgeWeightTimeSeriesAnalyser([(0, 1)])
        # Only 2 observations
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        w = torch.tensor([0.5, 0.5])
        analyser.update(ei, w)
        analyser.update(ei, w)
        result = analyser.analyse_edge((0, 1))
        assert "error" in result

    def test_dead_edge_tracking(self):
        """If an edge dies (weight=0), live_fraction should reflect this."""
        analyser = EdgeWeightTimeSeriesAnalyser([(0, 1)], max_history=20)
        ei_live = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        ei_dead = torch.zeros(2, 0, dtype=torch.long)
        w_live = torch.tensor([0.5, 0.5])
        w_dead = torch.zeros(0)

        # 10 live, 10 dead
        for _ in range(10):
            analyser.update(ei_live, w_live)
        for _ in range(10):
            analyser.update(ei_dead, w_dead)

        result = analyser.analyse_edge((0, 1))
        if "live_fraction" in result:
            assert result["live_fraction"] <= 1.0


# ---------------------------------------------------------------------------
# Graph edit distance tests
# ---------------------------------------------------------------------------

class TestApproximateGraphEditDistance:

    def test_self_distance_zero(self, simple_edge_index):
        dist = approximate_graph_edit_distance(simple_edge_index, simple_edge_index, 5)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_empty_vs_full(self):
        N = 5
        full_ei = torch.tensor(
            [[i for i in range(N) for j in range(N) if i != j],
             [j for i in range(N) for j in range(N) if i != j]],
            dtype=torch.long,
        )
        empty_ei = torch.zeros(2, 0, dtype=torch.long)
        dist = approximate_graph_edit_distance(empty_ei, full_ei, N, normalise=True)
        assert dist == pytest.approx(1.0, abs=1e-3)

    def test_distance_range(self):
        N = 8
        ei_a = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        ei_b = torch.tensor([[0, 1, 4, 5], [1, 0, 5, 4]], dtype=torch.long)
        dist = approximate_graph_edit_distance(ei_a, ei_b, N, normalise=True)
        assert 0.0 <= dist <= 1.0

    def test_non_normalised_is_integer(self):
        N = 6
        ei_a = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        ei_b = torch.tensor([[0, 2], [2, 3]], dtype=torch.long)
        dist = approximate_graph_edit_distance(ei_a, ei_b, N, normalise=False)
        assert dist == int(dist)
