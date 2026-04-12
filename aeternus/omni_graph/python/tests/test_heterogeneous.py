"""
tests/test_heterogeneous.py
============================
Tests for heterogeneous_graph module — multi-type financial graph GNNs.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from omni_graph.heterogeneous_graph import (
    FINANCIAL_NODE_TYPES,
    FINANCIAL_EDGE_TYPES,
    HeterogeneousFinancialGraphBuilder,
    TypeSpecificFeatureTransform,
    HANLayer,
    HANModel,
    HGTLayer,
    HGTModel,
    FinancialKGEmbedding,
    HeterogeneousGraphAggregator,
    HeterogeneousRegimeClassifier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_hetero_data():
    """Small returns datasets for heterogeneous graph construction."""
    np.random.seed(42)
    T = 60
    n_assets = 10
    n_sectors = 3
    n_macro = 4

    asset_returns = np.random.randn(T, n_assets).astype(np.float32)
    sector_returns = np.random.randn(T, n_sectors).astype(np.float32)
    macro_factors = np.random.randn(T, n_macro).astype(np.float32)

    asset_sector_map = {i: i % n_sectors for i in range(n_assets)}
    asset_exchange_map = {i: i % 2 for i in range(n_assets)}

    return {
        "asset_returns": asset_returns,
        "sector_returns": sector_returns,
        "macro_factors": macro_factors,
        "asset_sector_map": asset_sector_map,
        "asset_exchange_map": asset_exchange_map,
        "n_assets": n_assets,
        "n_sectors": n_sectors,
        "n_macro": n_macro,
        "n_exchanges": 2,
    }


@pytest.fixture
def node_types():
    return ["asset", "sector", "exchange", "macro_factor"]


@pytest.fixture
def edge_types():
    return [
        ("asset", "correlation", "asset"),
        ("asset", "belongs_to", "sector"),
        ("sector", "contains", "asset"),
        ("macro_factor", "influences", "asset"),
    ]


@pytest.fixture
def simple_x_dict():
    """Simple node feature dict."""
    return {
        "asset": torch.randn(10, 14),
        "sector": torch.randn(3, 4),
        "exchange": torch.randn(2, 4),
        "macro_factor": torch.randn(4, 4),
    }


@pytest.fixture
def simple_edge_index_dict():
    """Simple edge index dict."""
    return {
        ("asset", "correlation", "asset"): torch.tensor(
            [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long
        ),
        ("asset", "belongs_to", "sector"): torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 1, 1, 1, 2, 2, 2, 0]], dtype=torch.long
        ),
        ("sector", "contains", "asset"): torch.tensor(
            [[0, 0, 0, 1, 1, 1, 2, 2, 2, 0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long
        ),
        ("macro_factor", "influences", "asset"): torch.tensor(
            [[0, 0, 1, 1, 2, 2], [0, 5, 1, 6, 2, 7]], dtype=torch.long
        ),
    }


# ---------------------------------------------------------------------------
# HeterogeneousFinancialGraphBuilder tests
# ---------------------------------------------------------------------------

class TestHeterogeneousFinancialGraphBuilder:

    def test_build_returns_heterodata(self, small_hetero_data):
        try:
            from torch_geometric.data import HeteroData
        except ImportError:
            pytest.skip("torch_geometric not installed")

        builder = HeterogeneousFinancialGraphBuilder(corr_threshold=0.1)
        d = small_hetero_data
        data = builder.build(
            d["asset_returns"], d["sector_returns"], d["macro_factors"],
            d["asset_sector_map"], d["asset_exchange_map"], d["n_exchanges"],
        )
        assert isinstance(data, HeteroData)

    def test_asset_node_features(self, small_hetero_data):
        try:
            from torch_geometric.data import HeteroData
        except ImportError:
            pytest.skip("torch_geometric not installed")

        builder = HeterogeneousFinancialGraphBuilder(corr_threshold=0.1)
        d = small_hetero_data
        data = builder.build(
            d["asset_returns"], d["sector_returns"], d["macro_factors"],
            d["asset_sector_map"], d["asset_exchange_map"], d["n_exchanges"],
        )
        assert data["asset"].x.shape[0] == d["n_assets"]
        assert data["asset"].x.shape[1] == 4

    def test_macro_influence_edges(self, small_hetero_data):
        try:
            from torch_geometric.data import HeteroData
        except ImportError:
            pytest.skip("torch_geometric not installed")

        builder = HeterogeneousFinancialGraphBuilder(causal_threshold=0.0)
        d = small_hetero_data
        data = builder.build(
            d["asset_returns"], d["sector_returns"], d["macro_factors"],
            d["asset_sector_map"], d["asset_exchange_map"], d["n_exchanges"],
        )
        # With threshold=0, should have influence edges
        etype = ("macro_factor", "influences", "asset")
        if hasattr(data, etype):
            ei = data[etype].edge_index
            assert ei.shape[0] == 2

    def test_sector_membership_edges(self, small_hetero_data):
        try:
            from torch_geometric.data import HeteroData
        except ImportError:
            pytest.skip("torch_geometric not installed")

        builder = HeterogeneousFinancialGraphBuilder(corr_threshold=0.0)
        d = small_hetero_data
        data = builder.build(
            d["asset_returns"], d["sector_returns"], d["macro_factors"],
            d["asset_sector_map"], d["asset_exchange_map"], d["n_exchanges"],
        )
        etype = ("asset", "belongs_to", "sector")
        if hasattr(data, etype):
            ei = data[etype].edge_index
            assert ei.shape[1] == d["n_assets"]


# ---------------------------------------------------------------------------
# TypeSpecificFeatureTransform tests
# ---------------------------------------------------------------------------

class TestTypeSpecificFeatureTransform:

    def test_output_shape(self, simple_x_dict):
        feat_dims = {k: v.shape[-1] for k, v in simple_x_dict.items()}
        transform = TypeSpecificFeatureTransform(feat_dims, out_dim=32)
        out = transform(simple_x_dict)
        for ntype, x in out.items():
            assert x.shape[-1] == 32

    def test_all_types_transformed(self, simple_x_dict):
        feat_dims = {k: v.shape[-1] for k, v in simple_x_dict.items()}
        transform = TypeSpecificFeatureTransform(feat_dims, out_dim=64)
        out = transform(simple_x_dict)
        for ntype in simple_x_dict:
            assert ntype in out

    def test_unknown_type_handled(self, simple_x_dict):
        feat_dims = {k: v.shape[-1] for k, v in simple_x_dict.items()}
        transform = TypeSpecificFeatureTransform(feat_dims, out_dim=32)
        x_new = dict(simple_x_dict)
        x_new["unknown_type"] = torch.randn(5, 32)  # same dim as out
        out = transform(x_new)
        assert "unknown_type" in out

    def test_gradients_flow(self, simple_x_dict):
        feat_dims = {k: v.shape[-1] for k, v in simple_x_dict.items()}
        transform = TypeSpecificFeatureTransform(feat_dims, out_dim=16)
        x_grad = {k: v.clone().requires_grad_(True) for k, v in simple_x_dict.items()}
        out = transform(x_grad)
        loss = sum(o.sum() for o in out.values())
        loss.backward()
        for k, x in x_grad.items():
            assert x.grad is not None

    def test_output_finite(self, simple_x_dict):
        feat_dims = {k: v.shape[-1] for k, v in simple_x_dict.items()}
        transform = TypeSpecificFeatureTransform(feat_dims, out_dim=32)
        out = transform(simple_x_dict)
        for ntype, x in out.items():
            assert torch.isfinite(x).all()


# ---------------------------------------------------------------------------
# HANLayer tests
# ---------------------------------------------------------------------------

class TestHANLayer:

    def test_forward_output_keys(self, simple_x_dict, simple_edge_index_dict, node_types, edge_types):
        han = HANLayer(
            in_dim=32, out_dim=32,
            meta_paths=edge_types[:3],
            n_heads=2,
        )
        # First project to in_dim
        projected = {k: torch.randn(v.shape[0], 32) for k, v in simple_x_dict.items()}
        out = han(projected, simple_edge_index_dict)
        for ntype in projected:
            assert ntype in out

    def test_forward_output_shape(self, simple_x_dict, simple_edge_index_dict, edge_types):
        out_dim = 48
        han = HANLayer(in_dim=32, out_dim=out_dim, meta_paths=edge_types[:2])
        projected = {k: torch.randn(v.shape[0], 32) for k, v in simple_x_dict.items()}
        out = han(projected, simple_edge_index_dict)
        for ntype, h in out.items():
            assert h.shape[-1] == out_dim

    def test_han_with_empty_edge_type(self, simple_x_dict, edge_types):
        han = HANLayer(in_dim=16, out_dim=16, meta_paths=edge_types)
        projected = {k: torch.randn(v.shape[0], 16) for k, v in simple_x_dict.items()}
        # Provide only some edge types
        partial_ei_dict = {
            ("asset", "correlation", "asset"): torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        }
        out = han(projected, partial_ei_dict)
        assert len(out) > 0

    def test_han_no_nan(self, simple_x_dict, simple_edge_index_dict, edge_types):
        han = HANLayer(in_dim=32, out_dim=32, meta_paths=edge_types[:2])
        projected = {k: torch.randn(v.shape[0], 32) for k, v in simple_x_dict.items()}
        out = han(projected, simple_edge_index_dict)
        for ntype, h in out.items():
            assert torch.isfinite(h).all(), f"NaN in {ntype} output"


# ---------------------------------------------------------------------------
# HANModel tests
# ---------------------------------------------------------------------------

class TestHANModel:

    def test_han_model_forward(self, simple_x_dict, simple_edge_index_dict, edge_types):
        feat_dims = {k: v.shape[-1] for k, v in simple_x_dict.items()}
        model = HANModel(
            node_feat_dims=feat_dims,
            hidden_dim=32,
            out_dim=16,
            n_layers=2,
            n_heads=2,
            meta_paths=edge_types[:2],
        )
        out = model(simple_x_dict, simple_edge_index_dict)
        for ntype in feat_dims:
            if ntype in out:
                assert out[ntype].shape[-1] == 16

    def test_han_model_output_finite(self, simple_x_dict, simple_edge_index_dict, edge_types):
        feat_dims = {k: v.shape[-1] for k, v in simple_x_dict.items()}
        model = HANModel(
            node_feat_dims=feat_dims,
            hidden_dim=16,
            out_dim=8,
            meta_paths=edge_types[:2],
        )
        out = model(simple_x_dict, simple_edge_index_dict)
        for ntype, h in out.items():
            assert torch.isfinite(h).all()


# ---------------------------------------------------------------------------
# HGTLayer tests
# ---------------------------------------------------------------------------

class TestHGTLayer:

    def test_hgt_layer_forward(self, simple_x_dict, simple_edge_index_dict, node_types, edge_types):
        in_dim = 32
        projected = {k: torch.randn(v.shape[0], in_dim) for k, v in simple_x_dict.items()}
        layer = HGTLayer(
            in_dim=in_dim, out_dim=32,
            node_types=node_types, edge_types=edge_types[:3],
            n_heads=4,
        )
        out = layer(projected, simple_edge_index_dict)
        assert len(out) > 0

    def test_hgt_layer_output_shape(self, simple_x_dict, simple_edge_index_dict, node_types, edge_types):
        in_dim, out_dim = 32, 32
        projected = {k: torch.randn(v.shape[0], in_dim) for k, v in simple_x_dict.items()}
        layer = HGTLayer(
            in_dim=in_dim, out_dim=out_dim,
            node_types=node_types, edge_types=edge_types[:3],
            n_heads=4,
        )
        out = layer(projected, simple_edge_index_dict)
        for ntype, h in out.items():
            assert h.shape[-1] == out_dim

    def test_hgt_layer_no_nan(self, simple_x_dict, simple_edge_index_dict, node_types, edge_types):
        in_dim = 32
        projected = {k: torch.randn(v.shape[0], in_dim) for k, v in simple_x_dict.items()}
        layer = HGTLayer(
            in_dim=in_dim, out_dim=in_dim,
            node_types=node_types, edge_types=edge_types[:3],
            n_heads=4,
        )
        out = layer(projected, simple_edge_index_dict)
        for ntype, h in out.items():
            assert torch.isfinite(h).all()


# ---------------------------------------------------------------------------
# HGTModel tests
# ---------------------------------------------------------------------------

class TestHGTModel:

    def test_hgt_model_forward(self, simple_x_dict, simple_edge_index_dict, node_types, edge_types):
        feat_dims = {k: v.shape[-1] for k, v in simple_x_dict.items()}
        model = HGTModel(
            node_feat_dims=feat_dims,
            node_types=node_types,
            edge_types=edge_types[:3],
            hidden_dim=32,
            out_dim=16,
            n_layers=2,
            n_heads=4,
        )
        out = model(simple_x_dict, simple_edge_index_dict)
        for ntype in node_types:
            if ntype in out:
                assert out[ntype].shape[-1] == 16

    def test_hgt_gradient_flow(self, simple_x_dict, simple_edge_index_dict, node_types, edge_types):
        feat_dims = {k: v.shape[-1] for k, v in simple_x_dict.items()}
        model = HGTModel(
            node_feat_dims=feat_dims,
            node_types=node_types,
            edge_types=edge_types[:3],
            hidden_dim=16,
            out_dim=8,
        )
        x_grad = {k: v.clone().requires_grad_(True) for k, v in simple_x_dict.items()}
        out = model(x_grad, simple_edge_index_dict)
        loss = sum(h.sum() for h in out.values())
        loss.backward()
        # At least some gradients should flow
        has_grad = any(x.grad is not None for x in x_grad.values())
        assert has_grad

    def test_hgt_model_no_nan(self, simple_x_dict, simple_edge_index_dict, node_types, edge_types):
        feat_dims = {k: v.shape[-1] for k, v in simple_x_dict.items()}
        model = HGTModel(
            node_feat_dims=feat_dims,
            node_types=node_types,
            edge_types=edge_types[:3],
            hidden_dim=16,
            out_dim=8,
        )
        out = model(simple_x_dict, simple_edge_index_dict)
        for ntype, h in out.items():
            assert torch.isfinite(h).all(), f"NaN in HGT output for {ntype}"


# ---------------------------------------------------------------------------
# FinancialKGEmbedding tests
# ---------------------------------------------------------------------------

class TestFinancialKGEmbedding:

    def test_transe_score_shape(self):
        model = FinancialKGEmbedding(n_entities=20, n_relations=5, embed_dim=32, model="transe")
        B = 8
        h = torch.randint(0, 20, (B,))
        r = torch.randint(0, 5, (B,))
        t = torch.randint(0, 20, (B,))
        scores = model.score(h, r, t)
        assert scores.shape == (B,)

    def test_rotate_score_shape(self):
        model = FinancialKGEmbedding(n_entities=20, n_relations=5, embed_dim=32, model="rotate")
        B = 8
        h = torch.randint(0, 20, (B,))
        r = torch.randint(0, 5, (B,))
        t = torch.randint(0, 20, (B,))
        scores = model.score(h, r, t)
        assert scores.shape == (B,)

    def test_transe_scores_non_negative(self):
        """TransE scores (L1/L2 distances) should be non-negative."""
        model = FinancialKGEmbedding(n_entities=20, n_relations=5, embed_dim=32, model="transe")
        h = torch.randint(0, 20, (16,))
        r = torch.randint(0, 5, (16,))
        t = torch.randint(0, 20, (16,))
        scores = model.score(h, r, t)
        assert (scores >= 0).all()

    def test_margin_loss_shape(self):
        model = FinancialKGEmbedding(n_entities=20, n_relations=5, embed_dim=32)
        B = 8
        ph = torch.randint(0, 20, (B,))
        pr = torch.randint(0, 5, (B,))
        pt = torch.randint(0, 20, (B,))
        nh = torch.randint(0, 20, (B,))
        nr = torch.randint(0, 5, (B,))
        nt = torch.randint(0, 20, (B,))
        loss = model.margin_loss(ph, pr, pt, nh, nr, nt)
        assert loss.shape == ()  # scalar

    def test_margin_loss_non_negative(self):
        model = FinancialKGEmbedding(n_entities=20, n_relations=5, embed_dim=32)
        B = 4
        ph = torch.randint(0, 20, (B,))
        pr = torch.randint(0, 5, (B,))
        pt = torch.randint(0, 20, (B,))
        nh = torch.randint(0, 20, (B,))
        nr = torch.randint(0, 5, (B,))
        nt = torch.randint(0, 20, (B,))
        loss = model.margin_loss(ph, pr, pt, nh, nr, nt)
        assert float(loss) >= 0.0

    def test_embedding_extraction(self):
        n_e, n_r, d = 15, 4, 16
        model = FinancialKGEmbedding(n_e, n_r, d)
        embs = model.get_entity_embeddings()
        assert embs.shape == (n_e, d)
        rel_embs = model.get_relation_embeddings()
        assert rel_embs.shape == (n_r, d)

    def test_gradient_through_loss(self):
        model = FinancialKGEmbedding(n_entities=10, n_relations=3, embed_dim=16)
        B = 4
        ph = torch.randint(0, 10, (B,))
        pr = torch.randint(0, 3, (B,))
        pt = torch.randint(0, 10, (B,))
        nh = torch.randint(0, 10, (B,))
        nr = torch.randint(0, 3, (B,))
        nt = torch.randint(0, 10, (B,))
        loss = model.margin_loss(ph, pr, pt, nh, nr, nt)
        loss.backward()
        assert model.entity_embed.weight.grad is not None


# ---------------------------------------------------------------------------
# HeterogeneousGraphAggregator tests
# ---------------------------------------------------------------------------

class TestHeterogeneousGraphAggregator:

    def test_mean_pooling(self, node_types):
        agg = HeterogeneousGraphAggregator(node_types, node_dim=16, out_dim=8, pool="mean")
        x_dict = {nt: torch.randn(10, 16) for nt in node_types}
        out = agg(x_dict)
        assert out.shape == (8,)

    def test_max_pooling(self, node_types):
        agg = HeterogeneousGraphAggregator(node_types, node_dim=16, out_dim=8, pool="max")
        x_dict = {nt: torch.randn(5, 16) for nt in node_types}
        out = agg(x_dict)
        assert out.shape == (8,)

    def test_attention_pooling(self, node_types):
        agg = HeterogeneousGraphAggregator(node_types, node_dim=16, out_dim=8, pool="attention")
        x_dict = {nt: torch.randn(5, 16) for nt in node_types}
        out = agg(x_dict)
        assert out.shape == (8,)

    def test_output_finite(self, node_types):
        for pool in ("mean", "max", "attention", "sum"):
            agg = HeterogeneousGraphAggregator(node_types, node_dim=16, out_dim=8, pool=pool)
            x_dict = {nt: torch.randn(5, 16) for nt in node_types}
            out = agg(x_dict)
            assert torch.isfinite(out).all(), f"NaN with pool={pool}"

    def test_partial_types(self, node_types):
        """Only a subset of node types present → should not crash."""
        agg = HeterogeneousGraphAggregator(node_types, node_dim=16, out_dim=8)
        x_dict = {node_types[0]: torch.randn(5, 16)}
        out = agg(x_dict)
        assert out.shape == (8,)

    def test_empty_dict(self, node_types):
        agg = HeterogeneousGraphAggregator(node_types, node_dim=16, out_dim=8)
        out = agg({})
        assert isinstance(out, torch.Tensor)


# ---------------------------------------------------------------------------
# HeterogeneousRegimeClassifier tests
# ---------------------------------------------------------------------------

class TestHeterogeneousRegimeClassifier:

    def test_regime_classifier_forward(
        self, simple_x_dict, simple_edge_index_dict, node_types, edge_types
    ):
        feat_dims = {k: v.shape[-1] for k, v in simple_x_dict.items()}
        clf = HeterogeneousRegimeClassifier(
            node_feat_dims=feat_dims,
            node_types=node_types,
            edge_types=edge_types[:3],
            n_regimes=4,
            hidden_dim=32,
            embed_dim=16,
            n_hgt_layers=2,
            n_heads=4,
        )
        logits = clf(simple_x_dict, simple_edge_index_dict)
        assert logits.shape == (4,)

    def test_regime_classifier_output_valid(
        self, simple_x_dict, simple_edge_index_dict, node_types, edge_types
    ):
        feat_dims = {k: v.shape[-1] for k, v in simple_x_dict.items()}
        clf = HeterogeneousRegimeClassifier(
            node_feat_dims=feat_dims,
            node_types=node_types,
            edge_types=edge_types[:3],
            n_regimes=4,
            hidden_dim=16,
            embed_dim=8,
        )
        logits = clf(simple_x_dict, simple_edge_index_dict)
        assert torch.isfinite(logits).all()

    def test_predict_regime_valid_label(
        self, simple_x_dict, simple_edge_index_dict, node_types, edge_types
    ):
        feat_dims = {k: v.shape[-1] for k, v in simple_x_dict.items()}
        clf = HeterogeneousRegimeClassifier(
            node_feat_dims=feat_dims,
            node_types=node_types,
            edge_types=edge_types[:3],
            n_regimes=4,
            hidden_dim=16,
            embed_dim=8,
        )
        regime = clf.predict_regime(simple_x_dict, simple_edge_index_dict)
        assert 0 <= regime < 4

    def test_gradient_flow_through_classifier(
        self, simple_x_dict, simple_edge_index_dict, node_types, edge_types
    ):
        feat_dims = {k: v.shape[-1] for k, v in simple_x_dict.items()}
        clf = HeterogeneousRegimeClassifier(
            node_feat_dims=feat_dims,
            node_types=node_types,
            edge_types=edge_types[:3],
            n_regimes=4,
            hidden_dim=16,
            embed_dim=8,
        )
        clf.train()
        x_grad = {k: v.clone().requires_grad_(True) for k, v in simple_x_dict.items()}
        logits = clf(x_grad, simple_edge_index_dict)
        loss = logits.sum()
        loss.backward()
        has_grad = any(x.grad is not None for x in x_grad.values())
        assert has_grad


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestHeterogeneousIntegration:

    def test_full_pipeline_no_crash(self, small_hetero_data, node_types, edge_types):
        """Full pipeline: data → hetero graph → HGT → regime."""
        try:
            from torch_geometric.data import HeteroData
        except ImportError:
            pytest.skip("torch_geometric not installed")

        builder = HeterogeneousFinancialGraphBuilder(corr_threshold=0.1, causal_threshold=0.1)
        d = small_hetero_data
        data = builder.build(
            d["asset_returns"], d["sector_returns"], d["macro_factors"],
            d["asset_sector_map"], d["asset_exchange_map"], d["n_exchanges"],
        )

        # Build x_dict and edge_index_dict from HeteroData
        x_dict = {
            "asset": data["asset"].x,
            "sector": data["sector"].x,
            "exchange": data["exchange"].x,
            "macro_factor": data["macro_factor"].x,
        }
        feat_dims = {k: v.shape[-1] for k, v in x_dict.items()}

        edge_index_dict = {}
        for src, rel, dst in edge_types:
            try:
                ei = data[src, rel, dst].edge_index
                edge_index_dict[(src, rel, dst)] = ei
            except (KeyError, AttributeError):
                pass

        clf = HeterogeneousRegimeClassifier(
            node_feat_dims=feat_dims,
            node_types=node_types,
            edge_types=list(edge_index_dict.keys()),
            n_regimes=4,
            hidden_dim=32,
            embed_dim=16,
        )

        regime = clf.predict_regime(x_dict, edge_index_dict)
        assert 0 <= regime < 4

    def test_knowledge_graph_embedding_integration(self):
        """Test KGE + regime classifier pipeline."""
        n_entities = 15
        feat_dims = {"asset": 14, "sector": 4}
        node_types = ["asset", "sector"]
        edge_types = [("asset", "correlation", "asset")]

        clf = HeterogeneousRegimeClassifier(
            node_feat_dims=feat_dims,
            node_types=node_types,
            edge_types=edge_types,
            n_regimes=4,
            hidden_dim=16,
            embed_dim=8,
            use_kg_embed=True,
            n_entities=n_entities,
            n_relations=3,
        )

        x_dict = {
            "asset": torch.randn(10, 14),
            "sector": torch.randn(5, 4),
        }
        edge_index_dict = {
            ("asset", "correlation", "asset"): torch.tensor(
                [[0, 1, 2], [1, 2, 3]], dtype=torch.long
            )
        }
        entity_ids = torch.randint(0, n_entities, (10,))

        logits = clf(x_dict, edge_index_dict, entity_ids=entity_ids)
        assert logits.shape == (4,)
        assert torch.isfinite(logits).all()
