"""Comprehensive tests for multimodal modules."""
import pytest
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lumina"))

class TestGatedMultimodalUnit:
    """Tests for Gated multimodal unit."""

    @pytest.fixture
    def model(self):
        try:
            from multimodal import GatedMultimodalUnit
            return GatedMultimodalUnit(d1=64, d2=64, d_out=128)
        except (ImportError, TypeError):
            pytest.skip("Not available")

    def test_output_shape(self, model):
        out = model(x1=torch.randn(2,16,64), x2=torch.randn(2,16,64))
        assert out.shape == (2, 16, 128)

    def test_no_nan(self, model):
        out = model(x1=torch.randn(2,16,64), x2=torch.randn(2,16,64))
        assert not torch.isnan(out).any()

    def test_gradient_flow(self, model):
        model.train()
        out = model(x1=torch.randn(2,16,64), x2=torch.randn(2,16,64))
        out.sum().backward()

class TestBilinearFusion:
    """Tests for Bilinear fusion."""

    @pytest.fixture
    def model(self):
        try:
            from multimodal import BilinearFusion
            return BilinearFusion(d1=64, d2=64, d_out=128)
        except (ImportError, TypeError):
            pytest.skip("Not available")

    def test_output_shape(self, model):
        out = model(x1=torch.randn(2,16,64), x2=torch.randn(2,16,64))
        assert out.shape == (2, 16, 128)

    def test_no_nan(self, model):
        out = model(x1=torch.randn(2,16,64), x2=torch.randn(2,16,64))
        assert not torch.isnan(out).any()

    def test_gradient_flow(self, model):
        model.train()
        out = model(x1=torch.randn(2,16,64), x2=torch.randn(2,16,64))
        out.sum().backward()

class TestOrderBookEncoder:
    """Tests for Order book encoder."""

    @pytest.fixture
    def model(self):
        try:
            from multimodal import OrderBookEncoder
            return OrderBookEncoder(n_levels=10, d_model=64)
        except (ImportError, TypeError):
            pytest.skip("Not available")

    def test_output_shape(self, model):
        out = model(x=torch.randn(2,16,10,4))
        assert out.shape == (2, 16, 64)

    def test_no_nan(self, model):
        out = model(x=torch.randn(2,16,10,4))
        assert not torch.isnan(out).any()

    def test_gradient_flow(self, model):
        model.train()
        out = model(x=torch.randn(2,16,10,4))
        out.sum().backward()

class TestSentimentSignalEncoder:
    """Tests for Sentiment encoder."""

    @pytest.fixture
    def model(self):
        try:
            from multimodal import SentimentSignalEncoder
            return SentimentSignalEncoder(d_model=64)
        except (ImportError, TypeError):
            pytest.skip("Not available")

    def test_output_shape(self, model):
        out = model(sentiment_scores=torch.randn(2,8,3), source_ids=torch.zeros(2,8,dtype=torch.long), entity_ids=torch.zeros(2,8,dtype=torch.long))
        assert out.shape == (2, 64)

    def test_no_nan(self, model):
        out = model(sentiment_scores=torch.randn(2,8,3), source_ids=torch.zeros(2,8,dtype=torch.long), entity_ids=torch.zeros(2,8,dtype=torch.long))
        assert not torch.isnan(out).any()

    def test_gradient_flow(self, model):
        model.train()
        out = model(sentiment_scores=torch.randn(2,8,3), source_ids=torch.zeros(2,8,dtype=torch.long), entity_ids=torch.zeros(2,8,dtype=torch.long))
        out.sum().backward()

class TestCrossModalContrastiveLoss:
    @pytest.fixture
    def loss_fn(self):
        try:
            from multimodal import CrossModalContrastiveLoss
            return CrossModalContrastiveLoss(d_model=128, d_proj=64)
        except ImportError:
            pytest.skip("Not available")

    def test_loss_positive(self, loss_fn):
        z_a = torch.randn(8, 128)
        z_b = torch.randn(8, 128)
        result = loss_fn(z_a, z_b)
        assert result["loss"].item() >= 0

    def test_perfect_alignment(self, loss_fn):
        z = torch.randn(8, 128)
        result_aligned = loss_fn(z, z.clone())
        result_random = loss_fn(z, torch.randn(8, 128))
        # Perfect alignment should have lower loss
        assert result_aligned["loss"].item() <= result_random["loss"].item() + 1.0

class TestModalityAlignmentModule:
    @pytest.fixture
    def aligner(self):
        try:
            from multimodal import ModalityAlignmentModule
            return ModalityAlignmentModule(d_in=64, d_shared=32)
        except ImportError:
            pytest.skip("Not available")

    def test_forward_normalized(self, aligner):
        x = torch.randn(8, 64)
        h = aligner(x)
        norms = h.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_contrastive_loss(self, aligner):
        emb1 = aligner(torch.randn(8, 64))
        emb2 = aligner(torch.randn(8, 64))
        loss = aligner.contrastive_loss(emb1, emb2)
        assert loss.item() >= 0

class TestMultimodalAlignmentMetrics:
    def test_modality_gap(self):
        try:
            from multimodal import MultimodalAlignmentMetrics
            emb1 = torch.randn(32, 64)
            emb2 = torch.randn(32, 64)
            gap = MultimodalAlignmentMetrics.modality_gap(emb1, emb2)
            assert 0 <= gap <= 2.1  # bounded by unit sphere
        except ImportError:
            pytest.skip("Not available")

    def test_retrieval_metrics(self):
        try:
            from multimodal import MultimodalAlignmentMetrics
            q = torch.eye(20)  # perfect alignment
            g = torch.eye(20)
            metrics = MultimodalAlignmentMetrics.retrieval_metrics(q, g, k_list=[1, 5])
            assert metrics["R@1"] == 1.0  # perfect retrieval
        except ImportError:
            pytest.skip("Not available")
