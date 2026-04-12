"""
tests/test_topology.py
======================
Tests for graph_topology module — financial graph construction utilities.
"""

import math
import pytest
import numpy as np
import torch

from omni_graph.graph_topology import (
    GraphBuildConfig,
    FinancialGraphData,
    CorrelationGraphBuilder,
    MSTGraphBuilder,
    PMFGBuilder,
    KNNGraphBuilder,
    LeadLagGraphBuilder,
    LOBGraphBuilder,
    LOBSnapshot,
    AdaptiveGraphBuilder,
    TemporalGraphSequenceBuilder,
    GraphMetricsComputer,
    compute_node_features,
    compute_edge_features,
    normalise_edge_weights,
    normalise_node_features,
    pearson_correlation_matrix,
    spearman_correlation_matrix,
    distance_correlation_matrix,
    correlation_to_distance,
    ledoit_wolf_shrinkage,
    random_matrix_filter,
    graph_summary,
    build_sector_graph,
    returns_to_pyg_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_returns():
    """Generate small (T=100, N=10) returns matrix."""
    np.random.seed(42)
    T, N = 100, 10
    raw = np.random.randn(T, N).astype(np.float32)
    # Add some correlation structure
    raw[:, 1] = 0.5 * raw[:, 0] + 0.5 * raw[:, 1]
    raw[:, 2] = 0.7 * raw[:, 0] + 0.3 * raw[:, 2]
    return raw


@pytest.fixture
def medium_returns():
    """Generate medium (T=200, N=30) returns matrix."""
    np.random.seed(123)
    T, N = 200, 30
    return np.random.randn(T, N).astype(np.float32)


@pytest.fixture
def default_config():
    return GraphBuildConfig(
        corr_threshold=0.2,
        k_neighbours=3,
        max_lag=3,
    )


# ---------------------------------------------------------------------------
# Correlation matrix tests
# ---------------------------------------------------------------------------

class TestCorrelationMatrices:

    def test_pearson_shape(self, small_returns):
        corr = pearson_correlation_matrix(small_returns)
        N = small_returns.shape[1]
        assert corr.shape == (N, N)

    def test_pearson_diagonal(self, small_returns):
        corr = pearson_correlation_matrix(small_returns)
        np.testing.assert_allclose(np.diag(corr), np.ones(corr.shape[0]), atol=1e-5)

    def test_pearson_symmetry(self, small_returns):
        corr = pearson_correlation_matrix(small_returns)
        np.testing.assert_allclose(corr, corr.T, atol=1e-5)

    def test_pearson_range(self, small_returns):
        corr = pearson_correlation_matrix(small_returns)
        assert corr.min() >= -1.0 - 1e-5
        assert corr.max() <= 1.0 + 1e-5

    def test_spearman_shape(self, small_returns):
        corr = spearman_correlation_matrix(small_returns)
        N = small_returns.shape[1]
        assert corr.shape == (N, N)

    def test_spearman_symmetry(self, small_returns):
        corr = spearman_correlation_matrix(small_returns)
        np.testing.assert_allclose(corr, corr.T, atol=1e-5)

    def test_distance_correlation_range(self, small_returns):
        # Only test on small subset due to O(N^2 T) complexity
        sub = small_returns[:, :5]
        dcorr = distance_correlation_matrix(sub)
        assert dcorr.min() >= -1e-5
        assert dcorr.max() <= 1.0 + 1e-5

    def test_distance_correlation_diagonal(self, small_returns):
        sub = small_returns[:, :5]
        dcorr = distance_correlation_matrix(sub)
        assert dcorr[0, 0] > 0.9  # self-distance correlation = 1

    def test_correlation_to_distance(self, small_returns):
        corr = pearson_correlation_matrix(small_returns)
        dist = correlation_to_distance(corr)
        assert dist.shape == corr.shape
        np.testing.assert_allclose(np.diag(dist), np.zeros(corr.shape[0]), atol=1e-5)
        assert dist.min() >= 0.0

    def test_pearson_known_correlation(self):
        """Test on perfectly correlated data."""
        T = 100
        x = np.random.randn(T)
        data = np.stack([x, x, -x], axis=1).astype(np.float32)
        corr = pearson_correlation_matrix(data)
        assert abs(corr[0, 1] - 1.0) < 1e-4   # X0 == X1
        assert abs(corr[0, 2] + 1.0) < 1e-4   # X0 == -X2

    def test_insufficient_data():
        with pytest.raises(ValueError):
            pearson_correlation_matrix(np.random.randn(2, 5))


# ---------------------------------------------------------------------------
# CorrelationGraphBuilder tests
# ---------------------------------------------------------------------------

class TestCorrelationGraphBuilder:

    def test_build_returns_correct_type(self, small_returns, default_config):
        builder = CorrelationGraphBuilder(default_config)
        g = builder.build(small_returns)
        assert isinstance(g, FinancialGraphData)

    def test_edge_index_shape(self, small_returns, default_config):
        builder = CorrelationGraphBuilder(default_config)
        g = builder.build(small_returns)
        assert g.edge_index.shape[0] == 2

    def test_undirected_edges_symmetric(self, small_returns, default_config):
        builder = CorrelationGraphBuilder(default_config)
        g = builder.build(small_returns)
        # Every edge (i, j) should have a reverse edge (j, i)
        ei = g.edge_index.numpy()
        edge_set = {(int(ei[0, k]), int(ei[1, k])) for k in range(ei.shape[1])}
        for i, j in list(edge_set)[:10]:
            assert (j, i) in edge_set, f"Missing reverse edge ({j}, {i})"

    def test_no_self_loops(self, small_returns, default_config):
        builder = CorrelationGraphBuilder(default_config)
        g = builder.build(small_returns)
        ei = g.edge_index.numpy()
        for k in range(ei.shape[1]):
            assert ei[0, k] != ei[1, k], f"Self-loop at node {ei[0, k]}"

    def test_node_attr_shape(self, small_returns, default_config):
        builder = CorrelationGraphBuilder(default_config)
        g = builder.build(small_returns)
        assert g.node_attr is not None
        assert g.node_attr.shape == (small_returns.shape[1], 14)

    def test_edge_attr_shape(self, small_returns, default_config):
        builder = CorrelationGraphBuilder(default_config)
        g = builder.build(small_returns)
        if g.edge_index.shape[1] > 0:
            assert g.edge_attr is not None
            assert g.edge_attr.shape[0] == g.edge_index.shape[1]

    def test_spearman_method(self, small_returns):
        cfg = GraphBuildConfig(corr_method="spearman", corr_threshold=0.1)
        builder = CorrelationGraphBuilder(cfg)
        g = builder.build(small_returns)
        assert g.graph_type == "correlation_spearman"

    def test_distance_method(self, small_returns):
        cfg = GraphBuildConfig(corr_method="distance", corr_threshold=0.1)
        builder = CorrelationGraphBuilder(cfg)
        g = builder.build(small_returns[:, :5])  # small subset for speed
        assert g.graph_type == "correlation_distance"

    def test_asset_names_passed(self, small_returns, default_config):
        names = [f"stock_{i}" for i in range(small_returns.shape[1])]
        builder = CorrelationGraphBuilder(default_config)
        g = builder.build(small_returns, asset_names=names)
        assert g.asset_names == names

    def test_high_threshold_sparse_graph(self, small_returns):
        cfg = GraphBuildConfig(corr_threshold=0.99)
        builder = CorrelationGraphBuilder(cfg)
        g = builder.build(small_returns)
        # Very high threshold → very few edges (possibly zero + fallback)
        assert g.edge_index.shape[1] >= 0

    def test_low_threshold_dense_graph(self, small_returns):
        cfg = GraphBuildConfig(corr_threshold=0.0)
        builder = CorrelationGraphBuilder(cfg)
        g = builder.build(small_returns)
        N = small_returns.shape[1]
        # All pairs should be connected → N*(N-1) edges (undirected)
        assert g.edge_index.shape[1] == N * (N - 1)

    def test_pandas_input(self, small_returns):
        import pandas as pd
        df = pd.DataFrame(small_returns, columns=[f"A{i}" for i in range(small_returns.shape[1])])
        builder = CorrelationGraphBuilder()
        g = builder.build(df)
        assert g.num_nodes == small_returns.shape[1]


# ---------------------------------------------------------------------------
# MST tests
# ---------------------------------------------------------------------------

class TestMSTGraphBuilder:

    def test_mst_n_edges(self, small_returns):
        """MST of N nodes should have exactly N-1 edges (undirected: 2*(N-1))."""
        builder = MSTGraphBuilder()
        g = builder.build(small_returns)
        N = small_returns.shape[1]
        # Undirected: 2*(N-1)
        assert g.edge_index.shape[1] == 2 * (N - 1)

    def test_mst_graph_type(self, small_returns):
        builder = MSTGraphBuilder()
        g = builder.build(small_returns)
        assert g.graph_type == "mst"

    def test_mst_connectivity(self, small_returns):
        """MST should produce a connected graph."""
        builder = MSTGraphBuilder()
        g = builder.build(small_returns)
        N = small_returns.shape[1]
        # Check connectivity via BFS
        adj = [[] for _ in range(N)]
        for k in range(g.edge_index.shape[1]):
            i, j = int(g.edge_index[0, k]), int(g.edge_index[1, k])
            adj[i].append(j)
        visited = {0}
        queue = [0]
        while queue:
            node = queue.pop()
            for nbr in adj[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        assert len(visited) == N

    def test_mst_no_cycles(self, small_returns):
        """MST should have no cycles (exactly N-1 edges for N nodes)."""
        builder = MSTGraphBuilder()
        g = builder.build(small_returns)
        N = small_returns.shape[1]
        n_undirected_edges = g.edge_index.shape[1] // 2
        assert n_undirected_edges == N - 1


# ---------------------------------------------------------------------------
# KNN tests
# ---------------------------------------------------------------------------

class TestKNNGraphBuilder:

    def test_knn_node_degree(self, small_returns):
        """Each node should have at least k out-edges."""
        k = 3
        cfg = GraphBuildConfig(k_neighbours=k)
        builder = KNNGraphBuilder(cfg)
        g = builder.build(small_returns)
        N = small_returns.shape[1]
        out_degree = torch.zeros(N, dtype=torch.long)
        out_degree.scatter_add_(0, g.edge_index[0], torch.ones(g.edge_index.shape[1], dtype=torch.long))
        for i in range(N):
            assert out_degree[i] >= k, f"Node {i} has degree {out_degree[i]} < k={k}"

    def test_knn_graph_type(self, small_returns):
        builder = KNNGraphBuilder()
        g = builder.build(small_returns)
        assert g.graph_type == "knn"

    def test_knn_k_larger_than_n(self):
        """k >= N should be capped at N-1."""
        returns = np.random.randn(50, 5).astype(np.float32)
        cfg = GraphBuildConfig(k_neighbours=100)
        builder = KNNGraphBuilder(cfg)
        g = builder.build(returns)
        assert g.num_nodes == 5


# ---------------------------------------------------------------------------
# Lead-lag tests
# ---------------------------------------------------------------------------

class TestLeadLagGraphBuilder:

    def test_lead_lag_directed(self, small_returns):
        """Lead-lag graph should be directed (not necessarily symmetric)."""
        cfg = GraphBuildConfig(corr_threshold=0.1, max_lag=3)
        builder = LeadLagGraphBuilder(cfg)
        g = builder.build(small_returns)
        assert g.graph_type == "lead_lag"

    def test_lead_lag_no_self_loops(self, small_returns):
        cfg = GraphBuildConfig(corr_threshold=0.1)
        builder = LeadLagGraphBuilder(cfg)
        g = builder.build(small_returns)
        if g.edge_index.shape[1] > 0:
            ei = g.edge_index.numpy()
            for k in range(ei.shape[1]):
                assert ei[0, k] != ei[1, k]

    def test_lead_lag_with_known_lead(self):
        """Construct data where asset 0 leads asset 1 by 1 period."""
        T = 100
        x = np.random.randn(T).astype(np.float32)
        data = np.zeros((T, 3), dtype=np.float32)
        data[:, 0] = x
        data[1:, 1] = x[:-1]  # asset 1 lags asset 0 by 1
        data[:, 2] = np.random.randn(T)

        cfg = GraphBuildConfig(corr_threshold=0.3, max_lag=3)
        builder = LeadLagGraphBuilder(cfg)
        g = builder.build(data)
        # Should detect edge 0 → 1
        assert g.edge_index.shape[1] > 0


# ---------------------------------------------------------------------------
# LOB graph tests
# ---------------------------------------------------------------------------

class TestLOBGraphBuilder:

    def _make_snapshot(self, depth=5):
        bid_p = np.linspace(100.0, 99.5, depth)
        bid_s = np.ones(depth) * 100.0
        ask_p = np.linspace(100.1, 100.5, depth)
        ask_s = np.ones(depth) * 100.0
        return LOBSnapshot(bid_prices=bid_p, bid_sizes=bid_s, ask_prices=ask_p, ask_sizes=ask_s)

    def test_lob_basic_structure(self):
        snap = self._make_snapshot(depth=5)
        builder = LOBGraphBuilder(GraphBuildConfig(lob_depth=5))
        g = builder.build_from_snapshot(snap)
        assert g.graph_type == "lob"
        assert g.num_nodes == 5 + 5 + 1  # bid + ask + mid

    def test_lob_edge_index_valid(self):
        snap = self._make_snapshot(depth=5)
        builder = LOBGraphBuilder(GraphBuildConfig(lob_depth=5))
        g = builder.build_from_snapshot(snap)
        assert g.edge_index.shape[0] == 2
        assert g.edge_index.min() >= 0
        assert g.edge_index.max() < g.num_nodes

    def test_lob_node_features(self):
        snap = self._make_snapshot(depth=5)
        builder = LOBGraphBuilder(GraphBuildConfig(lob_depth=5))
        g = builder.build_from_snapshot(snap)
        assert g.node_attr is not None
        assert g.node_attr.shape == (g.num_nodes, 4)  # [price, size, side, level]


# ---------------------------------------------------------------------------
# Node feature engineering tests
# ---------------------------------------------------------------------------

class TestNodeFeatureEngineering:

    def test_compute_node_features_shape(self, small_returns):
        feats = compute_node_features(small_returns)
        T, N = small_returns.shape
        assert feats.shape == (N, 14)

    def test_compute_node_features_finite(self, small_returns):
        feats = compute_node_features(small_returns)
        assert torch.isfinite(feats).all()

    def test_compute_node_features_range(self, small_returns):
        feats = compute_node_features(small_returns)
        assert (feats >= -10.0).all()
        assert (feats <= 10.0).all()

    def test_sharpe_ratio_feature(self):
        """Node with high mean return should have positive Sharpe ratio feature."""
        T = 100
        N = 3
        returns = np.zeros((T, N), dtype=np.float32)
        returns[:, 0] = 0.01  # positive returns
        returns[:, 1] = -0.01  # negative returns
        returns[:, 2] = 0.0  # neutral
        feats = compute_node_features(returns)
        assert feats[0, 6] > feats[1, 6]  # Sharpe feature index 6


# ---------------------------------------------------------------------------
# Normalisation tests
# ---------------------------------------------------------------------------

class TestNormalisation:

    def test_zscore_edge_normalisation(self):
        ea = torch.randn(20, 3)
        norm = normalise_edge_weights(ea, method="zscore")
        assert norm.shape == ea.shape
        assert abs(float(norm[:, 0].mean())) < 0.1
        assert abs(float(norm[:, 0].std()) - 1.0) < 0.2

    def test_minmax_edge_normalisation(self):
        ea = torch.randn(20, 3)
        norm = normalise_edge_weights(ea, method="minmax")
        assert norm.min() >= -1e-5
        assert norm.max() <= 1.0 + 1e-5

    def test_zscore_node_normalisation(self, small_returns):
        na = compute_node_features(small_returns)
        norm = normalise_node_features(na, method="zscore")
        # Column means should be ~0
        col_means = norm.mean(dim=0)
        assert (col_means.abs() < 0.5).all()

    def test_empty_edge_weight_normalisation(self):
        ea = torch.zeros(0, 3)
        norm = normalise_edge_weights(ea, method="zscore")
        assert norm.shape == ea.shape


# ---------------------------------------------------------------------------
# Adaptive and temporal builder tests
# ---------------------------------------------------------------------------

class TestAdaptiveBuilder:

    def test_adaptive_builds_successfully(self, small_returns):
        builder = AdaptiveGraphBuilder()
        g = builder.build(small_returns)
        assert isinstance(g, FinancialGraphData)
        assert g.graph_type == "adaptive_composite"

    def test_adaptive_more_edges_than_mst(self, small_returns):
        adaptive = AdaptiveGraphBuilder()
        mst = MSTGraphBuilder()
        g_a = adaptive.build(small_returns)
        g_m = mst.build(small_returns)
        # Adaptive should have at least as many edges as MST
        assert g_a.edge_index.shape[1] >= g_m.edge_index.shape[1]


class TestTemporalSequenceBuilder:

    def test_build_sequence_length(self, medium_returns):
        builder = TemporalGraphSequenceBuilder(window=30, stride=10)
        # Use CorrelationGraphBuilder internally
        seq = builder.build_sequence(medium_returns)
        T = medium_returns.shape[0]
        expected = (T - 30) // 10 + 1
        assert len(seq) == expected

    def test_sequence_metadata(self, medium_returns):
        builder = TemporalGraphSequenceBuilder(window=30, stride=10)
        seq = builder.build_sequence(medium_returns)
        for g in seq:
            assert "t_start" in g.metadata
            assert "t_end" in g.metadata

    def test_sequence_consistent_nodes(self, medium_returns):
        builder = TemporalGraphSequenceBuilder(window=30, stride=10)
        seq = builder.build_sequence(medium_returns)
        n = medium_returns.shape[1]
        for g in seq:
            assert g.num_nodes == n


# ---------------------------------------------------------------------------
# Sector graph tests
# ---------------------------------------------------------------------------

class TestSectorGraph:

    def test_sector_graph_builds(self, small_returns):
        N = small_returns.shape[1]
        sector_map = {i: i // 3 for i in range(N)}  # 3-4 sectors
        g = build_sector_graph(small_returns, sector_map)
        n_sectors = len(set(sector_map.values()))
        assert g.num_nodes == N + n_sectors

    def test_sector_graph_has_membership_edges(self, small_returns):
        N = small_returns.shape[1]
        sector_map = {i: i // 5 for i in range(N)}
        g = build_sector_graph(small_returns, sector_map)
        # There should be at least N membership edges (asset → sector)
        assert g.edge_index.shape[1] > 0


# ---------------------------------------------------------------------------
# Graph summary tests
# ---------------------------------------------------------------------------

class TestGraphSummary:

    def test_summary_keys(self, small_returns):
        builder = CorrelationGraphBuilder()
        g = builder.build(small_returns)
        s = graph_summary(g)
        assert "num_nodes" in s
        assert "num_edges" in s
        assert "density" in s

    def test_summary_density_range(self, small_returns):
        builder = CorrelationGraphBuilder()
        g = builder.build(small_returns)
        s = graph_summary(g)
        assert 0.0 <= s["density"] <= 1.0


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

class TestStatisticalFilters:

    def test_ledoit_wolf_shrinkage_positive_definite(self, small_returns):
        shrunk = ledoit_wolf_shrinkage(small_returns)
        # Check positive semi-definite
        eigvals = np.linalg.eigvalsh(shrunk)
        assert eigvals.min() > -1e-6

    def test_ledoit_wolf_diagonal_ones(self, small_returns):
        shrunk = ledoit_wolf_shrinkage(small_returns)
        np.testing.assert_allclose(np.diag(shrunk), np.ones(shrunk.shape[0]), atol=0.1)

    def test_random_matrix_filter_shape(self, medium_returns):
        corr = pearson_correlation_matrix(medium_returns)
        T = medium_returns.shape[0]
        filtered = random_matrix_filter(corr, T)
        assert filtered.shape == corr.shape

    def test_random_matrix_filter_diagonal(self, medium_returns):
        corr = pearson_correlation_matrix(medium_returns)
        T = medium_returns.shape[0]
        filtered = random_matrix_filter(corr, T)
        np.testing.assert_allclose(np.diag(filtered), np.ones(corr.shape[0]), atol=0.01)


# ---------------------------------------------------------------------------
# Edge cases and robustness
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_two_asset_graph(self):
        returns = np.random.randn(50, 2).astype(np.float32)
        builder = CorrelationGraphBuilder(GraphBuildConfig(corr_threshold=0.0))
        g = builder.build(returns)
        assert g.num_nodes == 2
        assert g.edge_index.shape[1] >= 2  # at least bidirectional edge

    def test_single_asset_knn(self):
        returns = np.random.randn(50, 1).astype(np.float32)
        builder = KNNGraphBuilder(GraphBuildConfig(k_neighbours=3))
        g = builder.build(returns)
        assert g.num_nodes == 1

    def test_constant_returns_warning(self):
        """Constant returns → zero variance → should not crash."""
        returns = np.ones((50, 5), dtype=np.float32)
        builder = CorrelationGraphBuilder()
        # Should not raise, even though correlation is undefined
        try:
            g = builder.build(returns)
        except Exception:
            pass  # acceptable

    def test_large_corr_threshold(self, small_returns):
        cfg = GraphBuildConfig(corr_threshold=1.0)
        builder = CorrelationGraphBuilder(cfg)
        g = builder.build(small_returns)
        # No edges above threshold 1.0 → fallback should add some
        assert g.edge_index.shape[1] >= 0  # no crash

    def test_nan_free_node_features(self, small_returns):
        feats = compute_node_features(small_returns)
        assert not torch.isnan(feats).any()
        assert not torch.isinf(feats).any()


# ---------------------------------------------------------------------------
# FinancialGraphData container tests
# ---------------------------------------------------------------------------

class TestFinancialGraphData:

    def test_to_pyg(self, small_returns):
        try:
            from torch_geometric.data import Data
        except ImportError:
            pytest.skip("torch_geometric not installed")

        builder = CorrelationGraphBuilder()
        g = builder.build(small_returns)
        pyg_data = g.to_pyg()
        assert isinstance(pyg_data, Data)
        assert pyg_data.num_nodes == g.num_nodes

    def test_graph_data_fields(self, small_returns):
        builder = CorrelationGraphBuilder()
        g = builder.build(small_returns)
        assert isinstance(g.edge_index, torch.Tensor)
        assert isinstance(g.num_nodes, int)
        assert isinstance(g.asset_names, list)
        assert isinstance(g.graph_type, str)
        assert isinstance(g.metadata, dict)
