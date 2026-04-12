"""Tests for extra attention mechanisms in attention.py."""
import pytest
import torch

class TestCosineAttention:
    def setup_method(self):
        from attention import CosineAttention
        self.model = CosineAttention(d_model=64, num_heads=4)

    def test_forward_shape_small(self):
        x = torch.randn(2, 8, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert out.shape == (2, 8, 64)

    def test_forward_shape_larger(self):
        x = torch.randn(4, 32, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert out.shape == (4, 32, 64)

    def test_no_nan(self):
        x = torch.randn(2, 16, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert not torch.isnan(out).any()

    def test_gradient(self):
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        out.sum().backward()
        assert x.grad is not None

    def test_eval_mode_consistent(self):
        x = torch.randn(2, 8, 64)
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(x)
            out2 = self.model(x)
        if isinstance(out1, tuple): out1, out2 = out1[0], out2[0]
        assert torch.allclose(out1, out2)

    def test_state_dict_loadable(self):
        from attention import CosineAttention
        sd = self.model.state_dict()
        model2 = CosineAttention(d_model=64, num_heads=4)
        model2.load_state_dict(sd)

class TestRetentiveAttention:
    def setup_method(self):
        from attention import RetentiveAttention
        self.model = RetentiveAttention(d_model=64, num_heads=4)

    def test_forward_shape_small(self):
        x = torch.randn(2, 8, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert out.shape == (2, 8, 64)

    def test_forward_shape_larger(self):
        x = torch.randn(4, 32, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert out.shape == (4, 32, 64)

    def test_no_nan(self):
        x = torch.randn(2, 16, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert not torch.isnan(out).any()

    def test_gradient(self):
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        out.sum().backward()
        assert x.grad is not None

    def test_eval_mode_consistent(self):
        x = torch.randn(2, 8, 64)
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(x)
            out2 = self.model(x)
        if isinstance(out1, tuple): out1, out2 = out1[0], out2[0]
        assert torch.allclose(out1, out2)

    def test_state_dict_loadable(self):
        from attention import RetentiveAttention
        sd = self.model.state_dict()
        model2 = RetentiveAttention(d_model=64, num_heads=4)
        model2.load_state_dict(sd)

class TestMultiQueryAttention:
    def setup_method(self):
        from attention import MultiQueryAttention
        self.model = MultiQueryAttention(d_model=64, num_heads=4)

    def test_forward_shape_small(self):
        x = torch.randn(2, 8, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert out.shape == (2, 8, 64)

    def test_forward_shape_larger(self):
        x = torch.randn(4, 32, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert out.shape == (4, 32, 64)

    def test_no_nan(self):
        x = torch.randn(2, 16, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert not torch.isnan(out).any()

    def test_gradient(self):
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        out.sum().backward()
        assert x.grad is not None

    def test_eval_mode_consistent(self):
        x = torch.randn(2, 8, 64)
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(x)
            out2 = self.model(x)
        if isinstance(out1, tuple): out1, out2 = out1[0], out2[0]
        assert torch.allclose(out1, out2)

    def test_state_dict_loadable(self):
        from attention import MultiQueryAttention
        sd = self.model.state_dict()
        model2 = MultiQueryAttention(d_model=64, num_heads=4)
        model2.load_state_dict(sd)

class TestALiBiAttention:
    def setup_method(self):
        from attention import ALiBiAttention
        self.model = ALiBiAttention(d_model=64, num_heads=4)

    def test_forward_shape_small(self):
        x = torch.randn(2, 8, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert out.shape == (2, 8, 64)

    def test_forward_shape_larger(self):
        x = torch.randn(4, 32, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert out.shape == (4, 32, 64)

    def test_no_nan(self):
        x = torch.randn(2, 16, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert not torch.isnan(out).any()

    def test_gradient(self):
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        out.sum().backward()
        assert x.grad is not None

    def test_eval_mode_consistent(self):
        x = torch.randn(2, 8, 64)
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(x)
            out2 = self.model(x)
        if isinstance(out1, tuple): out1, out2 = out1[0], out2[0]
        assert torch.allclose(out1, out2)

    def test_state_dict_loadable(self):
        from attention import ALiBiAttention
        sd = self.model.state_dict()
        model2 = ALiBiAttention(d_model=64, num_heads=4)
        model2.load_state_dict(sd)

class TestRoPEAttention:
    def setup_method(self):
        from attention import RoPEAttention
        self.model = RoPEAttention(d_model=64, num_heads=4)

    def test_forward_shape_small(self):
        x = torch.randn(2, 8, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert out.shape == (2, 8, 64)

    def test_forward_shape_larger(self):
        x = torch.randn(4, 32, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert out.shape == (4, 32, 64)

    def test_no_nan(self):
        x = torch.randn(2, 16, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert not torch.isnan(out).any()

    def test_gradient(self):
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        out.sum().backward()
        assert x.grad is not None

    def test_eval_mode_consistent(self):
        x = torch.randn(2, 8, 64)
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(x)
            out2 = self.model(x)
        if isinstance(out1, tuple): out1, out2 = out1[0], out2[0]
        assert torch.allclose(out1, out2)

    def test_state_dict_loadable(self):
        from attention import RoPEAttention
        sd = self.model.state_dict()
        model2 = RoPEAttention(d_model=64, num_heads=4)
        model2.load_state_dict(sd)

class TestFlashAttentionSimulator:
    def setup_method(self):
        from attention import FlashAttentionSimulator
        self.model = FlashAttentionSimulator(d_model=64, num_heads=4)

    def test_forward_shape_small(self):
        x = torch.randn(2, 8, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert out.shape == (2, 8, 64)

    def test_forward_shape_larger(self):
        x = torch.randn(4, 32, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert out.shape == (4, 32, 64)

    def test_no_nan(self):
        x = torch.randn(2, 16, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert not torch.isnan(out).any()

    def test_gradient(self):
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        out.sum().backward()
        assert x.grad is not None

    def test_eval_mode_consistent(self):
        x = torch.randn(2, 8, 64)
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(x)
            out2 = self.model(x)
        if isinstance(out1, tuple): out1, out2 = out1[0], out2[0]
        assert torch.allclose(out1, out2)

    def test_state_dict_loadable(self):
        from attention import FlashAttentionSimulator
        sd = self.model.state_dict()
        model2 = FlashAttentionSimulator(d_model=64, num_heads=4)
        model2.load_state_dict(sd)

class TestXFormersStyleAttention:
    def setup_method(self):
        from attention import XFormersStyleAttention
        self.model = XFormersStyleAttention(d_model=64, num_heads=4)

    def test_forward_shape_small(self):
        x = torch.randn(2, 8, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert out.shape == (2, 8, 64)

    def test_forward_shape_larger(self):
        x = torch.randn(4, 32, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert out.shape == (4, 32, 64)

    def test_no_nan(self):
        x = torch.randn(2, 16, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert not torch.isnan(out).any()

    def test_gradient(self):
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        out.sum().backward()
        assert x.grad is not None

    def test_eval_mode_consistent(self):
        x = torch.randn(2, 8, 64)
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(x)
            out2 = self.model(x)
        if isinstance(out1, tuple): out1, out2 = out1[0], out2[0]
        assert torch.allclose(out1, out2)

    def test_state_dict_loadable(self):
        from attention import XFormersStyleAttention
        sd = self.model.state_dict()
        model2 = XFormersStyleAttention(d_model=64, num_heads=4)
        model2.load_state_dict(sd)

class TestKVCacheMultiHeadAttention:
    def setup_method(self):
        from attention import KVCacheMultiHeadAttention
        self.model = KVCacheMultiHeadAttention(d_model=64, num_heads=4)

    def test_forward_shape_small(self):
        x = torch.randn(2, 8, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert out.shape == (2, 8, 64)

    def test_forward_shape_larger(self):
        x = torch.randn(4, 32, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert out.shape == (4, 32, 64)

    def test_no_nan(self):
        x = torch.randn(2, 16, 64)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        assert not torch.isnan(out).any()

    def test_gradient(self):
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = self.model(x)
        if isinstance(out, tuple): out = out[0]
        out.sum().backward()
        assert x.grad is not None

    def test_eval_mode_consistent(self):
        x = torch.randn(2, 8, 64)
        self.model.eval()
        with torch.no_grad():
            out1 = self.model(x)
            out2 = self.model(x)
        if isinstance(out1, tuple): out1, out2 = out1[0], out2[0]
        assert torch.allclose(out1, out2)

    def test_state_dict_loadable(self):
        from attention import KVCacheMultiHeadAttention
        sd = self.model.state_dict()
        model2 = KVCacheMultiHeadAttention(d_model=64, num_heads=4)
        model2.load_state_dict(sd)

@pytest.mark.parametrize('d_model,num_heads,B,T', [
    (32, 4, 1, 8),
    (32, 4, 1, 16),
    (32, 4, 1, 32),
    (32, 4, 2, 8),
    (32, 4, 2, 16),
    (32, 4, 2, 32),
    (32, 8, 1, 8),
    (32, 8, 1, 16),
    (32, 8, 1, 32),
    (32, 8, 2, 8),
    (32, 8, 2, 16),
    (32, 8, 2, 32),
    (64, 4, 1, 8),
    (64, 4, 1, 16),
    (64, 4, 1, 32),
    (64, 4, 2, 8),
    (64, 4, 2, 16),
    (64, 4, 2, 32),
    (64, 8, 1, 8),
    (64, 8, 1, 16),
    (64, 8, 1, 32),
    (64, 8, 2, 8),
    (64, 8, 2, 16),
    (64, 8, 2, 32),
    (128, 4, 1, 8),
    (128, 4, 1, 16),
    (128, 4, 1, 32),
    (128, 4, 2, 8),
    (128, 4, 2, 16),
    (128, 4, 2, 32),
    (128, 8, 1, 8),
    (128, 8, 1, 16),
    (128, 8, 1, 32),
    (128, 8, 2, 8),
    (128, 8, 2, 16),
    (128, 8, 2, 32),
])
def test_rope_attention_parametrized(d_model, num_heads, B, T):
    from attention import RoPEAttention
    model = RoPEAttention(d_model, num_heads)
    x = torch.randn(B, T, d_model)
    out = model(x)
    assert out.shape == (B, T, d_model)
    assert not torch.isnan(out).any()

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
def test_alibi_attention_parametrized(d_model, num_heads, B, T):
    from attention import ALiBiAttention
    model = ALiBiAttention(d_model, num_heads)
    x = torch.randn(B, T, d_model)
    out = model(x)
    assert out.shape == (B, T, d_model)
    assert not torch.isnan(out).any()
