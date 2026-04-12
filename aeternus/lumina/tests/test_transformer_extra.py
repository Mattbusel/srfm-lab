"""Tests for extra transformer blocks."""
import pytest
import torch
import torch.nn as nn

class TestNormFormerBlock:
    def test_forward_shape(self):
        from transformer import NormFormerBlock
        block = NormFormerBlock(d_model=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == (2, 8, 64)

    def test_no_nan(self):
        from transformer import NormFormerBlock
        block = NormFormerBlock(d_model=64, num_heads=4)
        x = torch.randn(4, 16, 64)
        assert not torch.isnan(block(x)).any()

    def test_gradient(self):
        from transformer import NormFormerBlock
        block = NormFormerBlock(d_model=64, num_heads=4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None

    def test_state_dict(self):
        from transformer import NormFormerBlock
        b1 = NormFormerBlock(d_model=64, num_heads=4)
        b2 = NormFormerBlock(d_model=64, num_heads=4)
        b2.load_state_dict(b1.state_dict())
        x = torch.randn(2, 8, 64)
        assert torch.allclose(b1(x), b2(x))

class TestSandwichTransformerBlock:
    def test_forward_shape(self):
        from transformer import SandwichTransformerBlock
        block = SandwichTransformerBlock(d_model=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == (2, 8, 64)

    def test_no_nan(self):
        from transformer import SandwichTransformerBlock
        block = SandwichTransformerBlock(d_model=64, num_heads=4)
        x = torch.randn(4, 16, 64)
        assert not torch.isnan(block(x)).any()

    def test_gradient(self):
        from transformer import SandwichTransformerBlock
        block = SandwichTransformerBlock(d_model=64, num_heads=4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None

    def test_state_dict(self):
        from transformer import SandwichTransformerBlock
        b1 = SandwichTransformerBlock(d_model=64, num_heads=4)
        b2 = SandwichTransformerBlock(d_model=64, num_heads=4)
        b2.load_state_dict(b1.state_dict())
        x = torch.randn(2, 8, 64)
        assert torch.allclose(b1(x), b2(x))

class TestMacaronTransformerBlock:
    def test_forward_shape(self):
        from transformer import MacaronTransformerBlock
        block = MacaronTransformerBlock(d_model=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == (2, 8, 64)

    def test_no_nan(self):
        from transformer import MacaronTransformerBlock
        block = MacaronTransformerBlock(d_model=64, num_heads=4)
        x = torch.randn(4, 16, 64)
        assert not torch.isnan(block(x)).any()

    def test_gradient(self):
        from transformer import MacaronTransformerBlock
        block = MacaronTransformerBlock(d_model=64, num_heads=4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None

    def test_state_dict(self):
        from transformer import MacaronTransformerBlock
        b1 = MacaronTransformerBlock(d_model=64, num_heads=4)
        b2 = MacaronTransformerBlock(d_model=64, num_heads=4)
        b2.load_state_dict(b1.state_dict())
        x = torch.randn(2, 8, 64)
        assert torch.allclose(b1(x), b2(x))

class TestGatedTransformerBlock:
    def test_forward_shape(self):
        from transformer import GatedTransformerBlock
        block = GatedTransformerBlock(d_model=64)
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == (2, 8, 64)

    def test_no_nan(self):
        from transformer import GatedTransformerBlock
        block = GatedTransformerBlock(d_model=64)
        x = torch.randn(4, 16, 64)
        assert not torch.isnan(block(x)).any()

    def test_gradient(self):
        from transformer import GatedTransformerBlock
        block = GatedTransformerBlock(d_model=64)
        x = torch.randn(2, 8, 64, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None

    def test_state_dict(self):
        from transformer import GatedTransformerBlock
        b1 = GatedTransformerBlock(d_model=64)
        b2 = GatedTransformerBlock(d_model=64)
        b2.load_state_dict(b1.state_dict())
        x = torch.randn(2, 8, 64)
        assert torch.allclose(b1(x), b2(x))

class TestFNetBlock:
    def test_forward_shape(self):
        from transformer import FNetBlock
        block = FNetBlock(d_model=64)
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == (2, 8, 64)

    def test_no_nan(self):
        from transformer import FNetBlock
        block = FNetBlock(d_model=64)
        x = torch.randn(4, 16, 64)
        assert not torch.isnan(block(x)).any()

    def test_gradient(self):
        from transformer import FNetBlock
        block = FNetBlock(d_model=64)
        x = torch.randn(2, 8, 64, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None

    def test_state_dict(self):
        from transformer import FNetBlock
        b1 = FNetBlock(d_model=64)
        b2 = FNetBlock(d_model=64)
        b2.load_state_dict(b1.state_dict())
        x = torch.randn(2, 8, 64)
        assert torch.allclose(b1(x), b2(x))

class TestConformerBlock:
    def test_forward(self):
        from transformer import ConformerBlock
        block = ConformerBlock(d_model=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == (2, 8, 64)

    def test_no_nan(self):
        from transformer import ConformerBlock
        block = ConformerBlock(d_model=64, num_heads=4)
        x = torch.randn(4, 16, 64)
        assert not torch.isnan(block(x)).any()

class TestCrossAttentionBlock:
    def test_forward(self):
        from transformer import CrossAttentionBlock
        block = CrossAttentionBlock(d_model=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        mem = torch.randn(2, 4, 64)
        out = block(x, mem)
        assert out.shape == (2, 8, 64)

    def test_no_nan(self):
        from transformer import CrossAttentionBlock
        block = CrossAttentionBlock(d_model=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        mem = torch.randn(2, 6, 64)
        assert not torch.isnan(block(x, mem)).any()

    def test_gradient(self):
        from transformer import CrossAttentionBlock
        block = CrossAttentionBlock(d_model=64, num_heads=4)
        x = torch.randn(2, 8, 64, requires_grad=True)
        mem = torch.randn(2, 4, 64)
        block(x, mem).sum().backward()
        assert x.grad is not None

class TestStochasticDepthBlock:
    def test_forward_shape(self):
        from transformer import StochasticDepthBlock
        block = StochasticDepthBlock(d_model=64, num_heads=4, d_ff=256)
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == (2, 8, 64)

    def test_no_nan(self):
        from transformer import StochasticDepthBlock
        block = StochasticDepthBlock(d_model=64, num_heads=4, d_ff=256)
        x = torch.randn(4, 16, 64)
        assert not torch.isnan(block(x)).any()

    def test_gradient(self):
        from transformer import StochasticDepthBlock
        block = StochasticDepthBlock(d_model=64, num_heads=4, d_ff=256)
        x = torch.randn(2, 8, 64, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None

    def test_state_dict(self):
        from transformer import StochasticDepthBlock
        b1 = StochasticDepthBlock(d_model=64, num_heads=4, d_ff=256)
        b2 = StochasticDepthBlock(d_model=64, num_heads=4, d_ff=256)
        b2.load_state_dict(b1.state_dict())
        x = torch.randn(2, 8, 64)
        assert torch.allclose(b1(x), b2(x))

class TestTemporalTransformer:
    def test_forward_shape(self):
        from transformer import TemporalTransformer
        model = TemporalTransformer(d_model=32, num_heads=4, num_layers=2, max_len=64)
        x = torch.randn(2, 16)
        out = model(x)
        assert out.shape == (2, 16)

    def test_no_nan(self):
        from transformer import TemporalTransformer
        model = TemporalTransformer(d_model=32, num_heads=4, num_layers=2, max_len=64)
        x = torch.randn(4, 8)
        assert not torch.isnan(model(x)).any()

class TestUniversalTransformer:
    def test_forward_shape(self):
        from transformer import UniversalTransformer
        model = UniversalTransformer(d_model=32, num_heads=4, max_steps=3)
        x = torch.randn(2, 8, 32)
        out = model(x)
        assert out.shape == (2, 8, 32)

    def test_no_nan(self):
        from transformer import UniversalTransformer
        model = UniversalTransformer(d_model=32, num_heads=4, max_steps=3)
        x = torch.randn(2, 8, 32)
        assert not torch.isnan(model(x)).any()

class TestFNetBlock:
    def test_forward(self):
        from transformer import FNetBlock
        block = FNetBlock(64)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == (2, 16, 64)

class TestMLPMixerBlock:
    def test_forward(self):
        from transformer import MLPMixerBlock
        block = MLPMixerBlock(64, seq_len=8)
        x = torch.randn(2, 8, 64)
        out = block(x)
        assert out.shape == (2, 8, 64)

    def test_no_nan(self):
        from transformer import MLPMixerBlock
        block = MLPMixerBlock(32, seq_len=16)
        x = torch.randn(2, 16, 32)
        assert not torch.isnan(block(x)).any()

class TestHierarchicalTransformer:
    def test_forward(self):
        from transformer import HierarchicalTransformer
        model = HierarchicalTransformer(d_model=32, num_heads=4, local_window=4)
        x = torch.randn(2, 16, 32)
        out = model(x)
        assert out.shape == (2, 16, 32)

    def test_no_nan(self):
        from transformer import HierarchicalTransformer
        model = HierarchicalTransformer(d_model=32, num_heads=4, local_window=8)
        x = torch.randn(2, 32, 32)
        assert not torch.isnan(model(x)).any()

class TestConditionedTransformer:
    def test_cross_attn_conditioning(self):
        from transformer import ConditionedTransformer
        model = ConditionedTransformer(d_model=32, num_heads=4, d_context=16,
                                        num_layers=2, conditioning='cross_attn')
        x = torch.randn(2, 8, 32)
        ctx = torch.randn(2, 4, 16)
        out = model(x, ctx)
        assert out.shape == (2, 8, 32)

    def test_film_conditioning(self):
        from transformer import ConditionedTransformer
        model = ConditionedTransformer(d_model=32, num_heads=4, d_context=16,
                                        num_layers=2, conditioning='film')
        x = torch.randn(2, 8, 32)
        ctx = torch.randn(2, 16)
        out = model(x, ctx)
        assert out.shape == (2, 8, 32)

class TestStochasticDepthTransformer:
    def test_forward(self):
        from transformer import StochasticDepthTransformer
        model = StochasticDepthTransformer(d_model=32, num_heads=4, num_layers=4)
        x = torch.randn(2, 8, 32)
        out = model(x)
        assert out.shape == (2, 8, 32)

    def test_train_vs_eval(self):
        from transformer import StochasticDepthTransformer
        model = StochasticDepthTransformer(d_model=32, num_heads=4, num_layers=4,
                                            drop_path_rate=0.5)
        x = torch.randn(2, 8, 32)
        model.eval()
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

@pytest.mark.parametrize('d_model,num_heads,B,T', [
    (32, 4, 1, 8),
    (32, 4, 1, 16),
    (32, 4, 2, 8),
    (32, 4, 2, 16),
    (32, 8, 1, 8),
    (32, 8, 1, 16),
    (32, 8, 2, 8),
    (32, 8, 2, 16),
    (64, 4, 1, 8),
    (64, 4, 1, 16),
    (64, 4, 2, 8),
    (64, 4, 2, 16),
    (64, 8, 1, 8),
    (64, 8, 1, 16),
    (64, 8, 2, 8),
    (64, 8, 2, 16),
])
def test_normformer_parametrized(d_model, num_heads, B, T):
    from transformer import NormFormerBlock
    block = NormFormerBlock(d_model, num_heads)
    x = torch.randn(B, T, d_model)
    out = block(x)
    assert out.shape == (B, T, d_model)
    assert not torch.isnan(out).any()
