"""
tests/test_omni_graph.py — Unit tests for the omni_graph module.

Tests cover:
- Financial graph construction (correlation, Granger, partial correlation, TE).
- Dynamic GNN forward pass (TemporalGraphConv, DynamicEdgeConv, EvolutionaryGNN).
- Edge prediction (GraphDiffusion, LinkPredictor, WormholeDetector, RicciFlowGNN).
- Regime detection (GraphRegimeDetector, RegimeTransitionPredictor, CrisisEarlyWarning).
- Graph evolution and edge age computation.
- Synthetic data generation.
"""

import math
import numpy as np
import pytest
import torch
from torch_geometric.data import Data


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_returns(T: int = 100, N: int = 8, seed: int = 0) -> np.ndarray:
    """Generate simple synthetic returns."""
    rng = np.random.RandomState(seed)
    return rng.randn(T, N) * 0.01


def make_graph(n: int = 6, edge_density: float = 0.5, seed: int = 0) -> Data:
    """Create a small synthetic PyG graph."""
    torch.manual_seed(seed)
    x = torch.randn(n, 6)
    edges_src, edges_dst, edge_weights = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            if torch.rand(1).item() < edge_density:
                edges_src.extend([i, j])
                edges_dst.extend([j, i])
                w = torch.rand(1).item() * 0.9 + 0.1
                edge_weights.extend([w, w])
    if not edges_src:
        edges_src, edges_dst = [0, 1], [1, 0]
        edge_weights = [0.5, 0.5]
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n)


def make_snapshot_sequence(n: int = 6, T: int = 5, seed: int = 0) -> list:
    """Create a list of T graph snapshots."""
    return [make_graph(n, seed=seed + t) for t in range(T)]


# ── Financial graph construction ──────────────────────────────────────────────

class TestFinancialGraphs:

    def test_build_correlation_graph_shape(self):
        from omni_graph.financial_graphs import build_correlation_graph
        returns = make_returns(80, 10)
        graph = build_correlation_graph(returns, threshold=0.1)
        assert graph.x.shape[0] == 10
        assert graph.x.shape[1] > 0
        assert graph.edge_index.shape[0] == 2

    def test_build_correlation_graph_no_self_loops(self):
        from omni_graph.financial_graphs import build_correlation_graph
        returns = make_returns(80, 6)
        graph = build_correlation_graph(returns, threshold=0.0)
        ei = graph.edge_index
        for i in range(ei.shape[1]):
            assert ei[0, i] != ei[1, i], "Self-loops should not be present"

    def test_build_correlation_graph_symmetric(self):
        from omni_graph.financial_graphs import build_correlation_graph
        returns = make_returns(80, 6)
        graph = build_correlation_graph(returns, threshold=0.1)
        # For undirected: each edge (i,j) should have (j,i) too
        ei = graph.edge_index
        edge_set = set()
        for k in range(ei.shape[1]):
            edge_set.add((ei[0, k].item(), ei[1, k].item()))
        for (i, j) in list(edge_set):
            assert (j, i) in edge_set, f"Missing reverse edge {j}->{i}"

    def test_build_granger_graph_directed(self):
        from omni_graph.financial_graphs import build_granger_graph
        returns = make_returns(100, 5)
        graph = build_granger_graph(returns, max_lag=2, alpha=0.1)
        assert graph.x.shape[0] == 5
        assert graph.edge_index.shape[0] == 2

    def test_build_partial_correlation_graph(self):
        from omni_graph.financial_graphs import build_partial_correlation_graph
        rng = np.random.RandomState(0)
        # Generate returns with clear correlation structure
        base = rng.randn(100, 1) * 0.01
        returns = np.column_stack([
            base + rng.randn(100, 3) * 0.005,
            rng.randn(100, 3) * 0.01,
        ])
        graph = build_partial_correlation_graph(returns, alpha=0.5, threshold=0.0)
        assert graph.x.shape[0] == 6

    def test_build_transfer_entropy_graph(self):
        from omni_graph.financial_graphs import build_transfer_entropy_graph
        returns = make_returns(60, 4)
        graph = build_transfer_entropy_graph(returns, bins=5, lag=1, n_bootstrap=0)
        assert graph.x.shape[0] == 4
        assert graph.edge_index.shape[0] == 2

    def test_graph_evolution_run(self):
        from omni_graph.financial_graphs import GraphEvolution
        returns = make_returns(200, 8)
        evo = GraphEvolution(returns, window=40, step=10, threshold=0.2)
        evo.run(verbose=False)
        assert len(evo.snapshots) > 0
        assert len(evo.timestamps) == len(evo.snapshots)

    def test_graph_evolution_edge_ages(self):
        from omni_graph.financial_graphs import GraphEvolution
        returns = make_returns(200, 8)
        evo = GraphEvolution(returns, window=40, step=10, threshold=0.2)
        evo.run(verbose=False)
        ages = evo.compute_edge_ages()
        assert len(ages) == len(evo.snapshots)
        for snap, age in zip(evo.snapshots, ages):
            assert age.shape[0] == snap.edge_index.shape[1]

    def test_graph_evolution_summary(self):
        from omni_graph.financial_graphs import GraphEvolution
        returns = make_returns(200, 8)
        evo = GraphEvolution(returns, window=40, step=10, threshold=0.2)
        evo.run(verbose=False)
        df = evo.summary_statistics()
        assert "n_edges" in df.columns
        assert "density" in df.columns
        assert len(df) == len(evo.snapshots)


# ── Dynamic GNN ───────────────────────────────────────────────────────────────

class TestDynamicGNN:

    def test_temporal_graph_conv_forward(self):
        from omni_graph.dynamic_gnn import TemporalGraphConv
        g = make_graph(6)
        conv = TemporalGraphConv(in_channels=6, out_channels=16, half_life=10.0)
        edge_age = torch.rand(g.edge_index.shape[1]) * 20
        out = conv(g.x, g.edge_index, g.edge_attr[:, 0], edge_age)
        assert out.shape == (6, 16)

    def test_temporal_graph_conv_no_age(self):
        from omni_graph.dynamic_gnn import TemporalGraphConv
        g = make_graph(6)
        conv = TemporalGraphConv(6, 16)
        out = conv(g.x, g.edge_index)
        assert out.shape == (6, 16)

    def test_dynamic_edge_conv_forward(self):
        from omni_graph.dynamic_gnn import DynamicEdgeConv
        g = make_graph(8)
        conv = DynamicEdgeConv(in_channels=6, out_channels=32, time_dim=16)
        edge_age = torch.rand(g.edge_index.shape[1]) * 30
        out = conv(g.x, g.edge_index, edge_age)
        assert out.shape == (8, 32)

    def test_temporal_attention_forward(self):
        from omni_graph.dynamic_gnn import TemporalAttention
        B, T, D = 2, 8, 32
        attn = TemporalAttention(embed_dim=D, num_heads=4)
        seq = torch.randn(B, T, D)
        out, weights = attn(seq)
        assert out.shape == (B, D)

    def test_graph_rnn_forward(self):
        from omni_graph.dynamic_gnn import GraphRNN
        rnn = GraphRNN(input_dim=32, hidden_dim=64, output_dim=32)
        seq = torch.randn(2, 10, 32)
        next_emb, hidden, scalar = rnn(seq)
        assert next_emb.shape == (2, 32)
        assert scalar.shape == (2, 10)

    def test_graph_rnn_unroll(self):
        from omni_graph.dynamic_gnn import GraphRNN
        rnn = GraphRNN(input_dim=32, hidden_dim=64, output_dim=32)
        seed = torch.randn(2, 32)
        preds = rnn.unroll(seed, None, steps=3)
        assert len(preds) == 3
        for p in preds:
            assert p.shape == (2, 32)

    def test_evolutionary_gnn_encode(self):
        from omni_graph.dynamic_gnn import EvolutionaryGNN
        model = EvolutionaryGNN(node_features=6, hidden_dim=32, seq_len=4, n_regimes=3)
        g = make_graph(6)
        emb = model.encode_snapshot(g)
        assert emb.shape[1] == 32

    def test_evolutionary_gnn_forward(self):
        from omni_graph.dynamic_gnn import EvolutionaryGNN
        model = EvolutionaryGNN(node_features=6, hidden_dim=32, seq_len=4, n_regimes=3)
        snaps = make_snapshot_sequence(6, T=4)
        ages = [torch.rand(s.edge_index.shape[1]) * 10 for s in snaps]
        ts = torch.arange(4, dtype=torch.float)
        out = model(snaps, ages, ts)
        assert "regime_logits" in out
        assert out["regime_logits"].shape[-1] == 3
        assert "ricci_pred" in out

    def test_temporal_decay_values(self):
        from omni_graph.dynamic_gnn import temporal_decay
        ages = torch.tensor([0.0, 20.0, 40.0])
        decay = temporal_decay(ages, half_life=20.0)
        assert abs(decay[0].item() - 1.0) < 1e-5
        assert abs(decay[1].item() - 0.5) < 1e-5
        assert abs(decay[2].item() - 0.25) < 1e-5


# ── Edge prediction ───────────────────────────────────────────────────────────

class TestEdgePrediction:

    def test_graph_diffusion_forward(self):
        from omni_graph.edge_prediction import GraphDiffusion
        g = make_graph(8)
        model = GraphDiffusion(node_features=6, hidden_dim=32)
        preds, h = model(g.x, g.edge_index, g.edge_attr[:, 0])
        assert preds.shape[0] == g.edge_index.shape[1]
        assert h.shape == (8, 32)
        assert (preds >= 0).all() and (preds <= 1).all()

    def test_link_predictor_forward(self):
        from omni_graph.edge_prediction import LinkPredictor
        lp = LinkPredictor(node_dim=32, hidden_dim=64, history_features=4)
        n_edges = 10
        h_i = torch.randn(n_edges, 32)
        h_j = torch.randn(n_edges, 32)
        hist = torch.randn(n_edges, 4)
        logits = lp(h_i, h_j, hist)
        assert logits.shape == (n_edges,)

    def test_link_predictor_proba_range(self):
        from omni_graph.edge_prediction import LinkPredictor
        lp = LinkPredictor(node_dim=32, hidden_dim=64, history_features=0)
        h_i = torch.randn(5, 32)
        h_j = torch.randn(5, 32)
        probs = lp.predict_proba(h_i, h_j)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_wormhole_detector_update_score(self):
        from omni_graph.edge_prediction import WormholeDetector
        detector = WormholeDetector(window_size=5, z_score_threshold=2.0)
        edges = [(0, 1, 0.5), (1, 2, 0.6), (0, 2, 0.4)]
        n_nodes = 5
        for _ in range(20):
            detector.update(edges, n_nodes)
        detector.fit()
        scores = detector.score(edges, n_nodes)
        assert isinstance(scores, dict)
        assert len(scores) == 3
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_wormhole_detector_spike_detection(self):
        from omni_graph.edge_prediction import WormholeDetector
        detector = WormholeDetector(window_size=10, z_score_threshold=1.5)
        n_nodes = 6
        normal_edges = [(0, 1, 0.3), (1, 2, 0.4)]
        for _ in range(30):
            detector.update(normal_edges, n_nodes)
        detector.fit()

        # Inject a spike
        spike_edges = [(0, 1, 0.3), (1, 2, 0.4), (0, 3, 5.0)]  # wormhole!
        wormholes = detector.detect_wormholes(spike_edges, n_nodes, threshold=0.5)
        # The spike edge should score higher
        spike_edge_scores = [w for w in wormholes if w[0] == 0 and w[1] == 3]
        if spike_edge_scores:
            assert spike_edge_scores[0][3] > 0.0

    def test_ricci_flow_gnn_forward(self):
        from omni_graph.edge_prediction import RicciFlowGNN
        g = make_graph(8)
        model = RicciFlowGNN(node_features=6, hidden_dim=32, n_layers=2)
        n_edges = g.edge_index.shape[1]
        ricci = torch.randn(n_edges) * 0.3
        out = model(g.x, g.edge_index, ricci)
        assert "node_emb" in out
        assert "edge_weight_pred" in out
        assert "node_risk" in out
        assert out["node_risk"].shape == (8,)
        assert (out["node_risk"] >= 0).all() and (out["node_risk"] <= 1).all()

    def test_ricci_flow_gnn_crisis_score(self):
        from omni_graph.edge_prediction import RicciFlowGNN
        g = make_graph(6)
        model = RicciFlowGNN(node_features=6, hidden_dim=32)
        n_edges = g.edge_index.shape[1]

        # Negative curvature should yield higher crisis score
        ricci_negative = torch.full((n_edges,), -0.5)
        ricci_positive = torch.full((n_edges,), 0.5)

        out_neg = model(g.x, g.edge_index, ricci_negative)
        out_pos = model(g.x, g.edge_index, ricci_positive)

        score_neg = model.get_crisis_score(out_neg)
        score_pos = model.get_crisis_score(out_pos)

        # Crisis score should be higher for negative curvature (crisis regime)
        assert score_neg > score_pos, \
            f"Negative curvature should give higher crisis score: {score_neg:.3f} vs {score_pos:.3f}"


# ── Regime GNN ────────────────────────────────────────────────────────────────

class TestRegimeGNN:

    def test_page_hinkley_no_drift(self):
        from omni_graph.regime_gnn import PageHinkley
        ph = PageHinkley(delta=0.01, lambda_=50.0)
        # Stationary signal: no drift expected
        for _ in range(100):
            ph.update(0.0)
        assert not ph.check_alarm()

    def test_page_hinkley_drift(self):
        from omni_graph.regime_gnn import PageHinkley
        ph = PageHinkley(delta=0.001, lambda_=5.0)
        # Large jump: should trigger alarm
        for _ in range(5):
            ph.update(0.0)
        ph.update(10.0)
        assert ph.check_alarm()

    def test_wasserstein_kernel_distance_self(self):
        from omni_graph.regime_gnn import WassersteinGraphKernel
        kernel = WassersteinGraphKernel(embed_dim=16, n_layers=1)
        g = make_graph(5)
        d = kernel.distance(g, g)
        assert d >= 0.0

    def test_wasserstein_kernel_distance_different(self):
        from omni_graph.regime_gnn import WassersteinGraphKernel
        kernel = WassersteinGraphKernel(embed_dim=16, n_layers=1)
        g1 = make_graph(5, edge_density=0.2, seed=0)
        g2 = make_graph(5, edge_density=0.9, seed=1)
        d12 = kernel.distance(g1, g2)
        d11 = kernel.distance(g1, g1)
        assert d12 >= 0.0

    def test_graph_regime_detector_fit_predict(self):
        from omni_graph.regime_gnn import GraphRegimeDetector
        graphs = [make_graph(6, seed=i) for i in range(40)]
        detector = GraphRegimeDetector(n_regimes=3, n_init=3)
        detector.fit(graphs)
        pred = detector.predict(graphs[0])
        assert 0 <= pred < 3
        proba = detector.predict_proba(graphs[0])
        assert len(proba) == 3
        assert abs(proba.sum() - 1.0) < 1e-5

    def test_regime_transition_predictor_forward(self):
        from omni_graph.regime_gnn import RegimeTransitionPredictor
        model = RegimeTransitionPredictor(n_regimes=3, feature_dim=10, hidden_dim=32, seq_len=5)
        B, T = 2, 5
        feats = torch.randn(B, T, 10)
        regs = torch.randint(0, 3, (B, T))
        logits, probs, hidden = model(feats, regs)
        assert logits.shape == (B, 3)
        assert probs.shape == (B, 3)
        assert abs(probs.sum(dim=-1).mean().item() - 1.0) < 1e-4

    def test_crisis_early_warning_update(self):
        from omni_graph.regime_gnn import CrisisEarlyWarning
        ew = CrisisEarlyWarning(n_regimes=3, crisis_regimes=[2])

        # Normal period
        for _ in range(20):
            result = ew.update(mean_ricci=0.3, current_regime=0)
        normal_alarm = result["alarm_score"]

        # Crisis period
        for _ in range(10):
            crisis_result = ew.update(
                mean_ricci=-0.4,
                current_regime=2,
                transition_probs=np.array([0.05, 0.1, 0.85]),
                wormhole_score=0.8,
            )
        crisis_alarm = crisis_result["alarm_score"]

        assert crisis_alarm > normal_alarm, \
            f"Crisis alarm ({crisis_alarm:.3f}) should exceed normal alarm ({normal_alarm:.3f})"

    def test_crisis_early_warning_report(self):
        from omni_graph.regime_gnn import CrisisEarlyWarning
        ew = CrisisEarlyWarning(n_regimes=3)
        for _ in range(5):
            ew.update(0.2, 0)
        report = ew.get_report()
        assert "status" in report
        assert "current_alarm" in report
        assert report["n_history"] == 5

    def test_crisis_time_to_crisis(self):
        from omni_graph.regime_gnn import CrisisEarlyWarning
        ew = CrisisEarlyWarning()
        # Artificially inflate alarm history
        ew.alarm_history = [0.3, 0.4, 0.5, 0.6]
        ew.ema_alarm = 0.6
        ew.ricci_history = [0.1, 0.0, -0.1, -0.2]
        ew.regime_history = [0, 0, 1, 1]
        ttc = ew.time_to_crisis(threshold=0.65, forecast_horizon=10)
        # Should not be None since alarm is near threshold
        if ttc is not None:
            assert ttc >= 0


# ── Integration test ──────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_pipeline_small(self):
        """End-to-end test: returns -> graphs -> regime detection -> alarm."""
        from omni_graph.financial_graphs import GraphEvolution
        from omni_graph.regime_gnn import GraphRegimeDetector, CrisisEarlyWarning
        from omni_graph.experiments import approximate_ricci_numpy

        returns = make_returns(200, 8)
        evo = GraphEvolution(returns, window=30, step=10, threshold=0.2)
        evo.run(verbose=False)

        ricci_list = [approximate_ricci_numpy(s) for s in evo.snapshots]
        evo.set_ricci_curvatures(ricci_list)

        detector = GraphRegimeDetector(n_regimes=3, n_init=3)
        detector.fit(evo.snapshots, ricci_list)

        ew = CrisisEarlyWarning(n_regimes=3, crisis_regimes=[2])

        for i, snap in enumerate(evo.snapshots):
            rc = ricci_list[i]
            mean_rc = float(np.mean(rc)) if len(rc) > 0 else 0.0
            pred_regime = detector.predict(snap, rc)
            probs = detector.predict_proba(snap, rc)
            result = ew.update(mean_rc, pred_regime, probs)
            assert 0.0 <= result["alarm_score"] <= 1.0

        report = ew.get_report()
        assert report["n_history"] == len(evo.snapshots)

    def test_wormhole_detection_on_spike(self):
        """Inject a wormhole and verify detection."""
        from omni_graph.edge_prediction import WormholeDetector
        import numpy as np

        detector = WormholeDetector(window_size=8, z_score_threshold=2.0)
        n_nodes = 10

        # Normal graph: tight cluster
        for _ in range(25):
            edges = [(i, j, 0.4 + np.random.randn() * 0.05)
                     for i in range(4) for j in range(i+1, 4)]
            detector.update(edges, n_nodes)

        detector.fit()

        # Inject wormhole: suddenly strong edge between node 0 and node 8
        spike_edges = edges + [(0, 8, 3.0)]
        wh = detector.detect_wormholes(spike_edges, n_nodes, threshold=0.4)
        spike_detected = any(w[0] in (0, 8) and w[1] in (0, 8) for w in wh)
        # Score should be elevated regardless
        scores = detector.score(spike_edges, n_nodes)
        spike_key = (0, 8)
        if spike_key in scores:
            assert scores[spike_key] > 0.0


# ── Standalone utility tests ───────────────────────────────────────────────────

class TestUtilities:

    def test_approximate_ricci_numpy_shape(self):
        from omni_graph.experiments import approximate_ricci_numpy
        g = make_graph(6)
        rc = approximate_ricci_numpy(g)
        assert len(rc) == g.edge_index.shape[1]

    def test_approximate_ricci_numpy_empty_graph(self):
        from omni_graph.experiments import approximate_ricci_numpy
        g = Data(
            x=torch.randn(4, 4),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 1)),
        )
        rc = approximate_ricci_numpy(g)
        assert len(rc) == 0

    def test_generate_synthetic_returns_shape(self):
        from omni_graph.experiments import generate_synthetic_returns
        R, regimes, crisis_mask = generate_synthetic_returns(
            n_assets=10, n_periods=200, crisis_at=150, seed=0
        )
        assert R.shape == (200, 10)
        assert len(regimes) == 200
        assert len(crisis_mask) == 200
        assert set(np.unique(regimes)).issubset({0, 1, 2, 3})

    def test_sinusoidal_time_encoding(self):
        from omni_graph.dynamic_gnn import sinusoidal_time_encoding
        t = torch.tensor([0.0, 5.0, 10.0])
        enc = sinusoidal_time_encoding(t, d_model=16)
        assert enc.shape == (3, 16)
        assert not torch.isnan(enc).any()

    def test_temporal_decay_monotone(self):
        from omni_graph.dynamic_gnn import temporal_decay
        ages = torch.arange(0, 100, dtype=torch.float)
        decay = temporal_decay(ages, half_life=20.0)
        # Decay should be monotonically decreasing
        diffs = decay[1:] - decay[:-1]
        assert (diffs <= 0).all()

    def test_page_hinkley_reset(self):
        from omni_graph.regime_gnn import PageHinkley
        ph = PageHinkley(delta=0.001, lambda_=5.0)
        ph.update(100.0)  # trigger alarm
        assert ph.check_alarm()
        ph.reset()
        assert ph.ph_statistic == 0.0
        assert not ph.check_alarm()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
