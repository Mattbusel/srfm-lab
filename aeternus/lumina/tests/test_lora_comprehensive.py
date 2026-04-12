"""Comprehensive tests for lora.py - LoRA adaptation components."""
import pytest
import math
import torch
import torch.nn as nn
from unittest.mock import MagicMock

# ── Fixtures ──────────────────────────────────────────────────────────────

class TestLoRALinear:
    def setup_method(self):
        self.in_f = 64
        self.out_f = 128
        self.rank = 8

    def test_forward_shape(self):
        from lora import LoRALinear
        layer = LoRALinear(self.in_f, self.out_f, self.rank)
        x = torch.randn(2, 10, self.in_f)
        out = layer(x)
        assert out.shape == (2, 10, self.out_f)

    def test_no_nan(self):
        from lora import LoRALinear
        layer = LoRALinear(self.in_f, self.out_f, self.rank)
        x = torch.randn(4, 16, self.in_f)
        assert not torch.isnan(layer(x)).any()

    def test_gradient_flows(self):
        from lora import LoRALinear
        layer = LoRALinear(self.in_f, self.out_f, self.rank)
        x = torch.randn(2, 8, self.in_f)
        out = layer(x).sum()
        out.backward()
        assert layer.lora_A.grad is not None
        assert layer.lora_B.grad is not None

    def test_base_weight_frozen_when_required(self):
        from lora import LoRALinear
        layer = LoRALinear(self.in_f, self.out_f, self.rank)
        layer.weight.requires_grad_(False)
        x = torch.randn(2, 8, self.in_f)
        out = layer(x).sum()
        out.backward()
        assert layer.weight.grad is None

    def test_merge_unmerge(self):
        from lora import LoRALinear
        layer = LoRALinear(self.in_f, self.out_f, self.rank)
        x = torch.randn(2, 8, self.in_f)
        out_before = layer(x).detach().clone()
        layer.merge_weights()
        out_merged = layer(x).detach().clone()
        layer.unmerge_weights()
        out_after = layer(x).detach().clone()
        assert torch.allclose(out_before, out_merged, atol=1e-5)
        assert torch.allclose(out_before, out_after, atol=1e-5)

    def test_scaling(self):
        from lora import LoRALinear
        rank, alpha = 4, 8.0
        layer = LoRALinear(self.in_f, self.out_f, rank, alpha)
        assert layer.scaling == alpha / rank

    def test_no_bias(self):
        from lora import LoRALinear
        layer = LoRALinear(self.in_f, self.out_f, self.rank, bias=False)
        assert layer.bias is None
        x = torch.randn(2, 4, self.in_f)
        assert layer(x).shape == (2, 4, self.out_f)

    def test_dropout_effect_in_eval(self):
        from lora import LoRALinear
        layer = LoRALinear(self.in_f, self.out_f, self.rank, dropout=0.5)
        x = torch.randn(2, 8, self.in_f)
        layer.eval()
        out1 = layer(x)
        out2 = layer(x)
        assert torch.allclose(out1, out2)

    def test_extra_repr(self):
        from lora import LoRALinear
        layer = LoRALinear(self.in_f, self.out_f, self.rank)
        s = layer.extra_repr()
        assert 'rank' in s

class TestLoRAEmbedding:
    def test_forward_shape(self):
        from lora import LoRAEmbedding
        emb = LoRAEmbedding(1000, 64, rank=4)
        x = torch.randint(0, 1000, (2, 10))
        out = emb(x)
        assert out.shape == (2, 10, 64)

    def test_gradient(self):
        from lora import LoRAEmbedding
        emb = LoRAEmbedding(1000, 64, rank=4)
        x = torch.randint(0, 1000, (2, 10))
        emb(x).sum().backward()
        assert emb.lora_A.grad is not None

    def test_no_nan(self):
        from lora import LoRAEmbedding
        emb = LoRAEmbedding(500, 32, rank=8)
        x = torch.randint(0, 500, (4, 20))
        assert not torch.isnan(emb(x)).any()

class TestLoRAConv1d:
    def test_forward_shape(self):
        from lora import LoRAConv1d
        conv = LoRAConv1d(32, 64, kernel_size=3, rank=4, padding=1)
        x = torch.randn(2, 32, 50)
        out = conv(x)
        assert out.shape == (2, 64, 50)

    def test_gradient(self):
        from lora import LoRAConv1d
        conv = LoRAConv1d(16, 32, kernel_size=3, rank=4, padding=1)
        x = torch.randn(2, 16, 20)
        conv(x).sum().backward()
        assert conv.lora_A.grad is not None

class TestAdaLoRA:
    def test_forward_shape(self):
        from lora import AdaLoRA
        layer = AdaLoRA(64, 128, initial_rank=8, target_rank=4)
        x = torch.randn(2, 10, 64)
        out = layer(x)
        assert out.shape == (2, 10, 128)

    def test_no_nan(self):
        from lora import AdaLoRA
        layer = AdaLoRA(32, 64, initial_rank=6, target_rank=2)
        x = torch.randn(2, 8, 32)
        assert not torch.isnan(layer(x)).any()

    def test_prune_rank(self):
        from lora import AdaLoRA
        layer = AdaLoRA(64, 128, initial_rank=8, target_rank=4)
        layer.prune_to_rank(4)
        x = torch.randn(2, 5, 64)
        out = layer(x)
        assert out.shape == (2, 5, 128)

class TestVeRALayer:
    def test_forward(self):
        from lora import VeRALayer
        layer = VeRALayer(64, 128, rank=64)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        assert out.shape == (2, 8, 128)

    def test_few_trainable_params(self):
        from lora import VeRALayer
        layer = VeRALayer(64, 128, rank=64)
        trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        # vera_b(64) + vera_d(128) + weight(64*128) = very few vs full
        assert trainable < 64 * 128 * 2

class TestDoRALayer:
    def test_forward(self):
        from lora import DoRALayer
        layer = DoRALayer(64, 128, rank=8)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        assert out.shape == (2, 8, 128)

    def test_gradient(self):
        from lora import DoRALayer
        layer = DoRALayer(32, 64, rank=4)
        x = torch.randn(2, 4, 32)
        layer(x).sum().backward()
        assert layer.magnitude.grad is not None

class TestLoRAMoELayer:
    def test_forward(self):
        from lora import LoRAMoELayer
        layer = LoRAMoELayer(64, 128, num_experts=4, rank=4, top_k=2)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        assert out.shape == (2, 8, 128)

    def test_gradient(self):
        from lora import LoRAMoELayer
        layer = LoRAMoELayer(32, 64, num_experts=4, rank=4, top_k=2)
        x = torch.randn(2, 4, 32)
        layer(x).sum().backward()
        assert layer.lora_A.grad is not None

class TestQuantizedLoRALinear:
    def test_forward(self):
        from lora import QuantizedLoRALinear
        layer = QuantizedLoRALinear(64, 128, rank=4)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        assert out.shape == (2, 8, 128)

    def test_dequantize_shape(self):
        from lora import QuantizedLoRALinear
        layer = QuantizedLoRALinear(32, 64, rank=4, bits=8)
        w = layer.dequantize_weight()
        assert w.shape == (64, 32)

class TestMultiTaskLoRA:
    def test_forward_no_task(self):
        from lora import MultiTaskLoRA
        layer = MultiTaskLoRA(64, 128, num_tasks=4, rank=4)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        assert out.shape == (2, 8, 128)

    def test_forward_with_task(self):
        from lora import MultiTaskLoRA
        layer = MultiTaskLoRA(64, 128, num_tasks=4, rank=4)
        x = torch.randn(2, 8, 64)
        for task_id in range(4):
            out = layer(x, task_id=task_id)
            assert out.shape == (2, 8, 128)

class TestLoRAModelWrapper:
    def _make_model(self):
        return nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )

    def test_injection(self):
        from lora import LoRAModelWrapper, LoRAConfig
        config = LoRAConfig(rank=4, alpha=8.0, lora_on_all_linear=True)
        model = self._make_model()
        wrapper = LoRAModelWrapper(model, config)
        assert len(wrapper._lora_layers) > 0

    def test_fewer_trainable_params(self):
        from lora import LoRAModelWrapper, LoRAConfig
        config = LoRAConfig(rank=2, alpha=4.0, lora_on_all_linear=True)
        model = self._make_model()
        wrapper = LoRAModelWrapper(model, config)
        ratio = wrapper.trainable_ratio()
        assert ratio < 1.0

    def test_forward_works(self):
        from lora import LoRAModelWrapper, LoRAConfig
        config = LoRAConfig(rank=4, alpha=8.0, lora_on_all_linear=True)
        model = nn.Linear(32, 16)
        wrapper = LoRAModelWrapper(model, config)
        x = torch.randn(2, 10, 32)
        out = wrapper(x)
        assert out.shape == (2, 10, 16)

    def test_lora_state_dict_roundtrip(self):
        from lora import LoRAModelWrapper, LoRAConfig
        config = LoRAConfig(rank=4, alpha=8.0, lora_on_all_linear=True)
        model = nn.Linear(32, 16)
        wrapper = LoRAModelWrapper(model, config)
        sd = wrapper.lora_state_dict()
        wrapper.load_lora_state_dict(sd)

class TestLoRARegularizer:
    def test_zero_weights(self):
        from lora import LoRARegularizer, LoRALinear
        reg = LoRARegularizer(frobenius_weight=1e-4)
        layers = [LoRALinear(32, 64, rank=4) for _ in range(3)]
        loss = reg(layers)
        assert loss >= 0

    def test_nuclear_norm(self):
        from lora import LoRARegularizer, LoRALinear
        reg = LoRARegularizer(nuclear_weight=1e-4, frobenius_weight=0.0)
        layers = [LoRALinear(32, 64, rank=4)]
        loss = reg(layers)
        assert loss >= 0

# ── Parametrized LoRA shape tests ──────────────────────────────────────────
@pytest.mark.parametrize('in_f,out_f,rank,alpha,B,T', [
    (32, 64, 4, 8.0, 1, 8),
    (32, 64, 4, 8.0, 1, 16),
    (32, 64, 4, 8.0, 2, 8),
    (32, 64, 4, 8.0, 2, 16),
    (32, 64, 4, 16.0, 1, 8),
    (32, 64, 4, 16.0, 1, 16),
    (32, 64, 4, 16.0, 2, 8),
    (32, 64, 4, 16.0, 2, 16),
    (32, 64, 8, 8.0, 1, 8),
    (32, 64, 8, 8.0, 1, 16),
    (32, 64, 8, 8.0, 2, 8),
    (32, 64, 8, 8.0, 2, 16),
    (32, 64, 8, 16.0, 1, 8),
    (32, 64, 8, 16.0, 1, 16),
    (32, 64, 8, 16.0, 2, 8),
    (32, 64, 8, 16.0, 2, 16),
    (32, 128, 4, 8.0, 1, 8),
    (32, 128, 4, 8.0, 1, 16),
    (32, 128, 4, 8.0, 2, 8),
    (32, 128, 4, 8.0, 2, 16),
    (32, 128, 4, 16.0, 1, 8),
    (32, 128, 4, 16.0, 1, 16),
    (32, 128, 4, 16.0, 2, 8),
    (32, 128, 4, 16.0, 2, 16),
    (32, 128, 8, 8.0, 1, 8),
    (32, 128, 8, 8.0, 1, 16),
    (32, 128, 8, 8.0, 2, 8),
    (32, 128, 8, 8.0, 2, 16),
    (32, 128, 8, 16.0, 1, 8),
    (32, 128, 8, 16.0, 1, 16),
    (32, 128, 8, 16.0, 2, 8),
    (32, 128, 8, 16.0, 2, 16),
    (64, 64, 4, 8.0, 1, 8),
    (64, 64, 4, 8.0, 1, 16),
    (64, 64, 4, 8.0, 2, 8),
    (64, 64, 4, 8.0, 2, 16),
    (64, 64, 4, 16.0, 1, 8),
    (64, 64, 4, 16.0, 1, 16),
    (64, 64, 4, 16.0, 2, 8),
    (64, 64, 4, 16.0, 2, 16),
    (64, 64, 8, 8.0, 1, 8),
    (64, 64, 8, 8.0, 1, 16),
    (64, 64, 8, 8.0, 2, 8),
    (64, 64, 8, 8.0, 2, 16),
    (64, 64, 8, 16.0, 1, 8),
    (64, 64, 8, 16.0, 1, 16),
    (64, 64, 8, 16.0, 2, 8),
    (64, 64, 8, 16.0, 2, 16),
    (64, 128, 4, 8.0, 1, 8),
    (64, 128, 4, 8.0, 1, 16),
    (64, 128, 4, 8.0, 2, 8),
    (64, 128, 4, 8.0, 2, 16),
    (64, 128, 4, 16.0, 1, 8),
    (64, 128, 4, 16.0, 1, 16),
    (64, 128, 4, 16.0, 2, 8),
    (64, 128, 4, 16.0, 2, 16),
    (64, 128, 8, 8.0, 1, 8),
    (64, 128, 8, 8.0, 1, 16),
    (64, 128, 8, 8.0, 2, 8),
    (64, 128, 8, 8.0, 2, 16),
    (64, 128, 8, 16.0, 1, 8),
    (64, 128, 8, 16.0, 1, 16),
    (64, 128, 8, 16.0, 2, 8),
    (64, 128, 8, 16.0, 2, 16),
    (128, 64, 4, 8.0, 1, 8),
    (128, 64, 4, 8.0, 1, 16),
    (128, 64, 4, 8.0, 2, 8),
    (128, 64, 4, 8.0, 2, 16),
    (128, 64, 4, 16.0, 1, 8),
    (128, 64, 4, 16.0, 1, 16),
    (128, 64, 4, 16.0, 2, 8),
    (128, 64, 4, 16.0, 2, 16),
    (128, 64, 8, 8.0, 1, 8),
    (128, 64, 8, 8.0, 1, 16),
    (128, 64, 8, 8.0, 2, 8),
    (128, 64, 8, 8.0, 2, 16),
    (128, 64, 8, 16.0, 1, 8),
    (128, 64, 8, 16.0, 1, 16),
    (128, 64, 8, 16.0, 2, 8),
    (128, 64, 8, 16.0, 2, 16),
])
def test_lora_linear_parametrized(in_f, out_f, rank, alpha, B, T):
    from lora import LoRALinear
    layer = LoRALinear(in_f, out_f, rank, alpha)
    x = torch.randn(B, T, in_f)
    out = layer(x)
    assert out.shape == (B, T, out_f)
    assert not torch.isnan(out).any()
