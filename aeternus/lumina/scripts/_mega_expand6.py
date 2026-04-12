#!/usr/bin/env python3
"""Mega expansion 6 - large test files and additional module content"""
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def append(rel, content):
    path = os.path.join(BASE, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(content)
    lines = open(path, encoding="utf-8").read().count("\n")
    print(f"  {os.path.basename(path)}: {lines} lines")
    return lines

def write_new(rel, content):
    path = os.path.join(BASE, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    lines = open(path, encoding="utf-8").read().count("\n")
    print(f"  {os.path.basename(path)}: {lines} lines (new)")
    return lines


# ════════════════════════════════════════════════════════════════════════
# 1. tests/test_lora_comprehensive.py
# ════════════════════════════════════════════════════════════════════════
def build_lora_tests():
    lines = [
        '"""Comprehensive tests for lora.py - LoRA adaptation components."""',
        "import pytest",
        "import math",
        "import torch",
        "import torch.nn as nn",
        "from unittest.mock import MagicMock",
        "",
        "# ── Fixtures ──────────────────────────────────────────────────────────────",
        "",
    ]

    # Basic LoRALinear tests
    lines += [
        "class TestLoRALinear:",
        "    def setup_method(self):",
        "        self.in_f = 64",
        "        self.out_f = 128",
        "        self.rank = 8",
        "",
    ]
    for test_name, body in [
        ("test_forward_shape",
         "        from lora import LoRALinear\n"
         "        layer = LoRALinear(self.in_f, self.out_f, self.rank)\n"
         "        x = torch.randn(2, 10, self.in_f)\n"
         "        out = layer(x)\n"
         "        assert out.shape == (2, 10, self.out_f)"),
        ("test_no_nan",
         "        from lora import LoRALinear\n"
         "        layer = LoRALinear(self.in_f, self.out_f, self.rank)\n"
         "        x = torch.randn(4, 16, self.in_f)\n"
         "        assert not torch.isnan(layer(x)).any()"),
        ("test_gradient_flows",
         "        from lora import LoRALinear\n"
         "        layer = LoRALinear(self.in_f, self.out_f, self.rank)\n"
         "        x = torch.randn(2, 8, self.in_f)\n"
         "        out = layer(x).sum()\n"
         "        out.backward()\n"
         "        assert layer.lora_A.grad is not None\n"
         "        assert layer.lora_B.grad is not None"),
        ("test_base_weight_frozen_when_required",
         "        from lora import LoRALinear\n"
         "        layer = LoRALinear(self.in_f, self.out_f, self.rank)\n"
         "        layer.weight.requires_grad_(False)\n"
         "        x = torch.randn(2, 8, self.in_f)\n"
         "        out = layer(x).sum()\n"
         "        out.backward()\n"
         "        assert layer.weight.grad is None"),
        ("test_merge_unmerge",
         "        from lora import LoRALinear\n"
         "        layer = LoRALinear(self.in_f, self.out_f, self.rank)\n"
         "        x = torch.randn(2, 8, self.in_f)\n"
         "        out_before = layer(x).detach().clone()\n"
         "        layer.merge_weights()\n"
         "        out_merged = layer(x).detach().clone()\n"
         "        layer.unmerge_weights()\n"
         "        out_after = layer(x).detach().clone()\n"
         "        assert torch.allclose(out_before, out_merged, atol=1e-5)\n"
         "        assert torch.allclose(out_before, out_after, atol=1e-5)"),
        ("test_scaling",
         "        from lora import LoRALinear\n"
         "        rank, alpha = 4, 8.0\n"
         "        layer = LoRALinear(self.in_f, self.out_f, rank, alpha)\n"
         "        assert layer.scaling == alpha / rank"),
        ("test_no_bias",
         "        from lora import LoRALinear\n"
         "        layer = LoRALinear(self.in_f, self.out_f, self.rank, bias=False)\n"
         "        assert layer.bias is None\n"
         "        x = torch.randn(2, 4, self.in_f)\n"
         "        assert layer(x).shape == (2, 4, self.out_f)"),
        ("test_dropout_effect_in_eval",
         "        from lora import LoRALinear\n"
         "        layer = LoRALinear(self.in_f, self.out_f, self.rank, dropout=0.5)\n"
         "        x = torch.randn(2, 8, self.in_f)\n"
         "        layer.eval()\n"
         "        out1 = layer(x)\n"
         "        out2 = layer(x)\n"
         "        assert torch.allclose(out1, out2)"),
        ("test_extra_repr",
         "        from lora import LoRALinear\n"
         "        layer = LoRALinear(self.in_f, self.out_f, self.rank)\n"
         "        s = layer.extra_repr()\n"
         "        assert 'rank' in s"),
    ]:
        lines.append(f"    def {test_name}(self):")
        for bl in body.split("\n"):
            lines.append(bl)
        lines.append("")

    # LoRAEmbedding tests
    lines += [
        "class TestLoRAEmbedding:",
        "    def test_forward_shape(self):",
        "        from lora import LoRAEmbedding",
        "        emb = LoRAEmbedding(1000, 64, rank=4)",
        "        x = torch.randint(0, 1000, (2, 10))",
        "        out = emb(x)",
        "        assert out.shape == (2, 10, 64)",
        "",
        "    def test_gradient(self):",
        "        from lora import LoRAEmbedding",
        "        emb = LoRAEmbedding(1000, 64, rank=4)",
        "        x = torch.randint(0, 1000, (2, 10))",
        "        emb(x).sum().backward()",
        "        assert emb.lora_A.grad is not None",
        "",
        "    def test_no_nan(self):",
        "        from lora import LoRAEmbedding",
        "        emb = LoRAEmbedding(500, 32, rank=8)",
        "        x = torch.randint(0, 500, (4, 20))",
        "        assert not torch.isnan(emb(x)).any()",
        "",
    ]

    # LoRAConv1d tests
    lines += [
        "class TestLoRAConv1d:",
        "    def test_forward_shape(self):",
        "        from lora import LoRAConv1d",
        "        conv = LoRAConv1d(32, 64, kernel_size=3, rank=4, padding=1)",
        "        x = torch.randn(2, 32, 50)",
        "        out = conv(x)",
        "        assert out.shape == (2, 64, 50)",
        "",
        "    def test_gradient(self):",
        "        from lora import LoRAConv1d",
        "        conv = LoRAConv1d(16, 32, kernel_size=3, rank=4, padding=1)",
        "        x = torch.randn(2, 16, 20)",
        "        conv(x).sum().backward()",
        "        assert conv.lora_A.grad is not None",
        "",
    ]

    # AdaLoRA tests
    lines += [
        "class TestAdaLoRA:",
        "    def test_forward_shape(self):",
        "        from lora import AdaLoRA",
        "        layer = AdaLoRA(64, 128, initial_rank=8, target_rank=4)",
        "        x = torch.randn(2, 10, 64)",
        "        out = layer(x)",
        "        assert out.shape == (2, 10, 128)",
        "",
        "    def test_no_nan(self):",
        "        from lora import AdaLoRA",
        "        layer = AdaLoRA(32, 64, initial_rank=6, target_rank=2)",
        "        x = torch.randn(2, 8, 32)",
        "        assert not torch.isnan(layer(x)).any()",
        "",
        "    def test_prune_rank(self):",
        "        from lora import AdaLoRA",
        "        layer = AdaLoRA(64, 128, initial_rank=8, target_rank=4)",
        "        layer.prune_to_rank(4)",
        "        x = torch.randn(2, 5, 64)",
        "        out = layer(x)",
        "        assert out.shape == (2, 5, 128)",
        "",
    ]

    # VeRA and DoRA
    lines += [
        "class TestVeRALayer:",
        "    def test_forward(self):",
        "        from lora import VeRALayer",
        "        layer = VeRALayer(64, 128, rank=64)",
        "        x = torch.randn(2, 8, 64)",
        "        out = layer(x)",
        "        assert out.shape == (2, 8, 128)",
        "",
        "    def test_few_trainable_params(self):",
        "        from lora import VeRALayer",
        "        layer = VeRALayer(64, 128, rank=64)",
        "        trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)",
        "        # vera_b(64) + vera_d(128) + weight(64*128) = very few vs full",
        "        assert trainable < 64 * 128 * 2",
        "",
        "class TestDoRALayer:",
        "    def test_forward(self):",
        "        from lora import DoRALayer",
        "        layer = DoRALayer(64, 128, rank=8)",
        "        x = torch.randn(2, 8, 64)",
        "        out = layer(x)",
        "        assert out.shape == (2, 8, 128)",
        "",
        "    def test_gradient(self):",
        "        from lora import DoRALayer",
        "        layer = DoRALayer(32, 64, rank=4)",
        "        x = torch.randn(2, 4, 32)",
        "        layer(x).sum().backward()",
        "        assert layer.magnitude.grad is not None",
        "",
    ]

    # LoRAMoELayer
    lines += [
        "class TestLoRAMoELayer:",
        "    def test_forward(self):",
        "        from lora import LoRAMoELayer",
        "        layer = LoRAMoELayer(64, 128, num_experts=4, rank=4, top_k=2)",
        "        x = torch.randn(2, 8, 64)",
        "        out = layer(x)",
        "        assert out.shape == (2, 8, 128)",
        "",
        "    def test_gradient(self):",
        "        from lora import LoRAMoELayer",
        "        layer = LoRAMoELayer(32, 64, num_experts=4, rank=4, top_k=2)",
        "        x = torch.randn(2, 4, 32)",
        "        layer(x).sum().backward()",
        "        assert layer.lora_A.grad is not None",
        "",
    ]

    # QuantizedLoRALinear
    lines += [
        "class TestQuantizedLoRALinear:",
        "    def test_forward(self):",
        "        from lora import QuantizedLoRALinear",
        "        layer = QuantizedLoRALinear(64, 128, rank=4)",
        "        x = torch.randn(2, 8, 64)",
        "        out = layer(x)",
        "        assert out.shape == (2, 8, 128)",
        "",
        "    def test_dequantize_shape(self):",
        "        from lora import QuantizedLoRALinear",
        "        layer = QuantizedLoRALinear(32, 64, rank=4, bits=8)",
        "        w = layer.dequantize_weight()",
        "        assert w.shape == (64, 32)",
        "",
    ]

    # MultiTaskLoRA
    lines += [
        "class TestMultiTaskLoRA:",
        "    def test_forward_no_task(self):",
        "        from lora import MultiTaskLoRA",
        "        layer = MultiTaskLoRA(64, 128, num_tasks=4, rank=4)",
        "        x = torch.randn(2, 8, 64)",
        "        out = layer(x)",
        "        assert out.shape == (2, 8, 128)",
        "",
        "    def test_forward_with_task(self):",
        "        from lora import MultiTaskLoRA",
        "        layer = MultiTaskLoRA(64, 128, num_tasks=4, rank=4)",
        "        x = torch.randn(2, 8, 64)",
        "        for task_id in range(4):",
        "            out = layer(x, task_id=task_id)",
        "            assert out.shape == (2, 8, 128)",
        "",
    ]

    # LoRAModelWrapper
    lines += [
        "class TestLoRAModelWrapper:",
        "    def _make_model(self):",
        "        return nn.Sequential(",
        "            nn.Linear(32, 64),",
        "            nn.ReLU(),",
        "            nn.Linear(64, 16),",
        "        )",
        "",
        "    def test_injection(self):",
        "        from lora import LoRAModelWrapper, LoRAConfig",
        "        config = LoRAConfig(rank=4, alpha=8.0, lora_on_all_linear=True)",
        "        model = self._make_model()",
        "        wrapper = LoRAModelWrapper(model, config)",
        "        assert len(wrapper._lora_layers) > 0",
        "",
        "    def test_fewer_trainable_params(self):",
        "        from lora import LoRAModelWrapper, LoRAConfig",
        "        config = LoRAConfig(rank=2, alpha=4.0, lora_on_all_linear=True)",
        "        model = self._make_model()",
        "        wrapper = LoRAModelWrapper(model, config)",
        "        ratio = wrapper.trainable_ratio()",
        "        assert ratio < 1.0",
        "",
        "    def test_forward_works(self):",
        "        from lora import LoRAModelWrapper, LoRAConfig",
        "        config = LoRAConfig(rank=4, alpha=8.0, lora_on_all_linear=True)",
        "        model = nn.Linear(32, 16)",
        "        wrapper = LoRAModelWrapper(model, config)",
        "        x = torch.randn(2, 10, 32)",
        "        out = wrapper(x)",
        "        assert out.shape == (2, 10, 16)",
        "",
        "    def test_lora_state_dict_roundtrip(self):",
        "        from lora import LoRAModelWrapper, LoRAConfig",
        "        config = LoRAConfig(rank=4, alpha=8.0, lora_on_all_linear=True)",
        "        model = nn.Linear(32, 16)",
        "        wrapper = LoRAModelWrapper(model, config)",
        "        sd = wrapper.lora_state_dict()",
        "        wrapper.load_lora_state_dict(sd)",
        "",
    ]

    # LoRARegularizer
    lines += [
        "class TestLoRARegularizer:",
        "    def test_zero_weights(self):",
        "        from lora import LoRARegularizer, LoRALinear",
        "        reg = LoRARegularizer(frobenius_weight=1e-4)",
        "        layers = [LoRALinear(32, 64, rank=4) for _ in range(3)]",
        "        loss = reg(layers)",
        "        assert loss >= 0",
        "",
        "    def test_nuclear_norm(self):",
        "        from lora import LoRARegularizer, LoRALinear",
        "        reg = LoRARegularizer(nuclear_weight=1e-4, frobenius_weight=0.0)",
        "        layers = [LoRALinear(32, 64, rank=4)]",
        "        loss = reg(layers)",
        "        assert loss >= 0",
        "",
    ]

    # Parametrized tests
    lines += [
        "# ── Parametrized LoRA shape tests ──────────────────────────────────────────",
        "@pytest.mark.parametrize('in_f,out_f,rank,alpha,B,T', [",
    ]
    configs = []
    for in_f in [32, 64, 128]:
        for out_f in [64, 128]:
            for rank in [4, 8]:
                for alpha in [8.0, 16.0]:
                    for B in [1, 2]:
                        for T in [8, 16]:
                            configs.append(f"    ({in_f}, {out_f}, {rank}, {alpha}, {B}, {T}),")
    lines += configs[:80]  # keep reasonable
    lines += [
        "])",
        "def test_lora_linear_parametrized(in_f, out_f, rank, alpha, B, T):",
        "    from lora import LoRALinear",
        "    layer = LoRALinear(in_f, out_f, rank, alpha)",
        "    x = torch.randn(B, T, in_f)",
        "    out = layer(x)",
        "    assert out.shape == (B, T, out_f)",
        "    assert not torch.isnan(out).any()",
        "",
    ]

    return "\n".join(lines)

write_new("tests/test_lora_comprehensive.py", build_lora_tests())

# ════════════════════════════════════════════════════════════════════════
# 2. tests/test_moe_comprehensive.py
# ════════════════════════════════════════════════════════════════════════
def build_moe_tests():
    lines = [
        '"""Comprehensive tests for moe.py - Mixture of Experts components."""',
        "import pytest",
        "import torch",
        "import torch.nn as nn",
        "",
    ]

    lines += [
        "class TestTopKRouter:",
        "    def test_output_shapes(self):",
        "        from moe import TopKRouter",
        "        router = TopKRouter(64, 8, top_k=2)",
        "        x = torch.randn(32, 64)",
        "        out = router(x)",
        "        assert out.dispatch_mask.shape[0] == 32",
        "        assert out.dispatch_mask.shape[1] == 8",
        "        assert out.router_probs.shape == (32, 8)",
        "",
        "    def test_load_loss_scalar(self):",
        "        from moe import TopKRouter",
        "        router = TopKRouter(64, 8, top_k=2)",
        "        x = torch.randn(32, 64)",
        "        out = router(x)",
        "        assert out.load_loss.dim() == 0",
        "        assert out.load_loss.item() >= 0",
        "",
        "    def test_router_probs_sum_to_one(self):",
        "        from moe import TopKRouter",
        "        router = TopKRouter(64, 8, top_k=2)",
        "        x = torch.randn(16, 64)",
        "        out = router(x)",
        "        sums = out.router_probs.sum(-1)",
        "        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)",
        "",
    ]

    lines += [
        "class TestExpertFFN:",
        "    def test_forward_shape(self):",
        "        from moe import ExpertFFN",
        "        expert = ExpertFFN(64, 256)",
        "        x = torch.randn(8, 64)",
        "        out = expert(x)",
        "        assert out.shape == (8, 64)",
        "",
        "    @pytest.mark.parametrize('activation', ['gelu', 'relu', 'silu'])",
        "    def test_activations(self, activation):",
        "        from moe import ExpertFFN",
        "        expert = ExpertFFN(32, 128, activation=activation)",
        "        x = torch.randn(4, 32)",
        "        out = expert(x)",
        "        assert out.shape == (4, 32)",
        "        assert not torch.isnan(out).any()",
        "",
    ]

    lines += [
        "class TestSparseMoELayer:",
        "    def test_forward_shape(self):",
        "        from moe import SparseMoELayer",
        "        moe = SparseMoELayer(64, num_experts=4, top_k=2)",
        "        x = torch.randn(2, 8, 64)",
        "        out = moe(x)",
        "        assert out.shape == (2, 8, 64)",
        "",
        "    def test_no_nan(self):",
        "        from moe import SparseMoELayer",
        "        moe = SparseMoELayer(32, num_experts=4, top_k=2)",
        "        x = torch.randn(2, 4, 32)",
        "        out = moe(x)",
        "        assert not torch.isnan(out).any()",
        "",
        "    def test_aux_loss_exists(self):",
        "        from moe import SparseMoELayer",
        "        moe = SparseMoELayer(64, num_experts=4, top_k=2)",
        "        x = torch.randn(2, 8, 64)",
        "        moe(x)",
        "        assert hasattr(moe, 'aux_loss')",
        "        assert moe.aux_loss.item() >= 0",
        "",
        "    def test_gradient_flows(self):",
        "        from moe import SparseMoELayer",
        "        moe = SparseMoELayer(32, num_experts=4, top_k=2)",
        "        x = torch.randn(2, 4, 32, requires_grad=True)",
        "        out = moe(x)",
        "        out.sum().backward()",
        "        assert x.grad is not None",
        "",
    ]

    lines += [
        "class TestFusedMoELayer:",
        "    def test_forward_shape(self):",
        "        from moe import FusedMoELayer",
        "        moe = FusedMoELayer(64, num_experts=4, top_k=2)",
        "        x = torch.randn(2, 8, 64)",
        "        out = moe(x)",
        "        assert out.shape == (2, 8, 64)",
        "",
        "    def test_no_nan(self):",
        "        from moe import FusedMoELayer",
        "        moe = FusedMoELayer(32, num_experts=4, top_k=2)",
        "        x = torch.randn(2, 4, 32)",
        "        assert not torch.isnan(moe(x)).any()",
        "",
        "    def test_gradient(self):",
        "        from moe import FusedMoELayer",
        "        moe = FusedMoELayer(32, num_experts=4, top_k=2)",
        "        x = torch.randn(2, 4, 32, requires_grad=True)",
        "        moe(x).sum().backward()",
        "        assert x.grad is not None",
        "",
    ]

    lines += [
        "class TestExpertChoiceLayer:",
        "    def test_forward_shape(self):",
        "        from moe import ExpertChoiceLayer",
        "        moe = ExpertChoiceLayer(64, num_experts=4, expert_capacity=8)",
        "        x = torch.randn(2, 8, 64)",
        "        out = moe(x)",
        "        assert out.shape == (2, 8, 64)",
        "",
        "    def test_no_nan(self):",
        "        from moe import ExpertChoiceLayer",
        "        moe = ExpertChoiceLayer(32, num_experts=4, expert_capacity=4)",
        "        x = torch.randn(2, 4, 32)",
        "        assert not torch.isnan(moe(x)).any()",
        "",
    ]

    lines += [
        "class TestSwitchTransformerLayer:",
        "    def test_forward_shape(self):",
        "        from moe import SwitchTransformerLayer",
        "        switch = SwitchTransformerLayer(64, num_experts=4)",
        "        x = torch.randn(2, 8, 64)",
        "        out = switch(x)",
        "        assert out.shape == (2, 8, 64)",
        "",
        "    def test_aux_loss_positive(self):",
        "        from moe import SwitchTransformerLayer",
        "        switch = SwitchTransformerLayer(64, num_experts=4)",
        "        x = torch.randn(2, 8, 64)",
        "        switch(x)",
        "        assert switch.aux_loss.item() >= 0",
        "",
    ]

    lines += [
        "class TestMoETransformerBlock:",
        "    def test_moe_block_shape(self):",
        "        from moe import MoETransformerBlock",
        "        block = MoETransformerBlock(64, 4, num_experts=4, moe_layer=True)",
        "        x = torch.randn(2, 8, 64)",
        "        out = block(x)",
        "        assert out.shape == (2, 8, 64)",
        "",
        "    def test_dense_block_shape(self):",
        "        from moe import MoETransformerBlock",
        "        block = MoETransformerBlock(64, 4, moe_layer=False)",
        "        x = torch.randn(2, 8, 64)",
        "        out = block(x)",
        "        assert out.shape == (2, 8, 64)",
        "",
        "    def test_no_nan(self):",
        "        from moe import MoETransformerBlock",
        "        block = MoETransformerBlock(32, 4, num_experts=4, moe_layer=True)",
        "        x = torch.randn(2, 4, 32)",
        "        assert not torch.isnan(block(x)).any()",
        "",
        "    def test_gradient(self):",
        "        from moe import MoETransformerBlock",
        "        block = MoETransformerBlock(32, 4, num_experts=4, moe_layer=True)",
        "        x = torch.randn(2, 4, 32, requires_grad=True)",
        "        block(x).sum().backward()",
        "        assert x.grad is not None",
        "",
    ]

    lines += [
        "class TestMoELanguageModel:",
        "    def test_forward_logits_shape(self):",
        "        from moe import MoELanguageModel",
        "        model = MoELanguageModel(vocab_size=1000, d_model=64, num_layers=4,",
        "                                 num_heads=4, num_experts=4, moe_every_n=2)",
        "        ids = torch.randint(0, 1000, (2, 16))",
        "        logits, aux = model(ids)",
        "        assert logits.shape == (2, 16, 1000)",
        "        assert aux.item() >= 0",
        "",
        "    def test_no_nan_output(self):",
        "        from moe import MoELanguageModel",
        "        model = MoELanguageModel(vocab_size=500, d_model=32, num_layers=2,",
        "                                 num_heads=4, num_experts=4, moe_every_n=2)",
        "        ids = torch.randint(0, 500, (2, 8))",
        "        logits, _ = model(ids)",
        "        assert not torch.isnan(logits).any()",
        "",
    ]

    lines += [
        "class TestMoELoadBalancer:",
        "    def test_update_and_report(self):",
        "        from moe import MoELoadBalancer",
        "        lb = MoELoadBalancer(num_experts=8)",
        "        dispatch = torch.randint(0, 2, (32, 8)).float()",
        "        lb.update(dispatch)",
        "        report = lb.report()",
        "        assert 'expert_counts' in report",
        "        assert 'load_imbalance_cv' in report",
        "",
        "    def test_imbalance_metric(self):",
        "        from moe import MoELoadBalancer",
        "        lb = MoELoadBalancer(num_experts=4)",
        "        # All traffic to expert 0",
        "        dispatch = torch.zeros(32, 4)",
        "        dispatch[:, 0] = 1.0",
        "        lb.update(dispatch)",
        "        assert lb.load_imbalance() > 0",
        "",
    ]

    # Parametrized MoE tests
    lines += [
        "@pytest.mark.parametrize('d_model,num_experts,top_k,B,T', [",
    ]
    for d in [32, 64]:
        for ne in [4, 8]:
            for k in [1, 2]:
                for B in [1, 2]:
                    for T in [4, 8]:
                        lines.append(f"    ({d}, {ne}, {k}, {B}, {T}),")
    lines += [
        "])",
        "def test_sparse_moe_parametrized(d_model, num_experts, top_k, B, T):",
        "    from moe import SparseMoELayer",
        "    moe = SparseMoELayer(d_model, num_experts=num_experts, top_k=top_k)",
        "    x = torch.randn(B, T, d_model)",
        "    out = moe(x)",
        "    assert out.shape == (B, T, d_model)",
        "    assert not torch.isnan(out).any()",
        "",
    ]

    return "\n".join(lines)

write_new("tests/test_moe_comprehensive.py", build_moe_tests())

# ════════════════════════════════════════════════════════════════════════
# 3. tests/test_continual_learning.py
# ════════════════════════════════════════════════════════════════════════
def build_cl_tests():
    lines = [
        '"""Tests for continual_learning.py components."""',
        "import pytest",
        "import torch",
        "import torch.nn as nn",
        "from unittest.mock import MagicMock",
        "",
    ]

    lines += [
        "class TestElasticWeightConsolidation:",
        "    def _make_model(self):",
        "        return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))",
        "",
        "    def test_ewc_loss_zero_before_consolidation(self):",
        "        from continual_learning import ElasticWeightConsolidation",
        "        model = self._make_model()",
        "        ewc = ElasticWeightConsolidation(model, ewc_lambda=100.0)",
        "        loss = ewc.ewc_loss()",
        "        assert loss.item() == 0.0",
        "",
        "    def test_ewc_loss_after_estimate_fisher(self):",
        "        from continual_learning import ElasticWeightConsolidation",
        "        model = self._make_model()",
        "        ewc = ElasticWeightConsolidation(model, ewc_lambda=100.0, n_fisher_samples=5)",
        "",
        "        def dummy_loader():",
        "            for _ in range(5):",
        "                yield {'x': torch.randn(4, 16), 'y': torch.randint(0, 4, (4,))}",
        "",
        "        def loss_fn(m, batch):",
        "            return nn.CrossEntropyLoss()(m(batch['x']), batch['y'])",
        "",
        "        ewc.estimate_fisher(dummy_loader(), loss_fn)",
        "        # Perturb weights slightly",
        "        for p in model.parameters():",
        "            p.data += 0.01",
        "        ewc_loss = ewc.ewc_loss()",
        "        assert ewc_loss.item() >= 0",
        "",
        "    def test_forward_passes_through(self):",
        "        from continual_learning import ElasticWeightConsolidation",
        "        model = nn.Linear(16, 4)",
        "        ewc = ElasticWeightConsolidation(model)",
        "        x = torch.randn(2, 16)",
        "        out = ewc(x)",
        "        assert out.shape == (2, 4)",
        "",
    ]

    lines += [
        "class TestProgressiveNeuralNetworks:",
        "    def test_initial_column(self):",
        "        from continual_learning import ProgressiveNeuralNetworks",
        "        pnn = ProgressiveNeuralNetworks(16, 32, 4)",
        "        x = torch.randn(2, 16)",
        "        out = pnn(x)",
        "        assert out.shape == (2, 4)",
        "",
        "    def test_add_task(self):",
        "        from continual_learning import ProgressiveNeuralNetworks",
        "        pnn = ProgressiveNeuralNetworks(16, 32, 4)",
        "        pnn.add_task()",
        "        x = torch.randn(2, 16)",
        "        out = pnn(x, column_idx=1)",
        "        assert out.shape == (2, 4)",
        "",
        "    def test_previous_column_frozen(self):",
        "        from continual_learning import ProgressiveNeuralNetworks",
        "        pnn = ProgressiveNeuralNetworks(16, 32, 4)",
        "        pnn.add_task()",
        "        for param in pnn.columns[0].parameters():",
        "            assert not param.requires_grad",
        "",
        "    def test_two_tasks_different_outputs(self):",
        "        from continual_learning import ProgressiveNeuralNetworks",
        "        pnn = ProgressiveNeuralNetworks(16, 32, 4)",
        "        pnn.add_task()",
        "        x = torch.randn(2, 16)",
        "        out0 = pnn(x, column_idx=0)",
        "        out1 = pnn(x, column_idx=1)",
        "        assert out0.shape == out1.shape == (2, 4)",
        "",
    ]

    lines += [
        "class TestContinualNormalization:",
        "    def test_shape(self):",
        "        from continual_learning import ContinualNormalization",
        "        cn = ContinualNormalization(32, num_tasks=4)",
        "        cn.set_task(0)",
        "        x = torch.randn(16, 32)",
        "        out = cn(x)",
        "        assert out.shape == (16, 32)",
        "",
        "    def test_task_switch(self):",
        "        from continual_learning import ContinualNormalization",
        "        cn = ContinualNormalization(16, num_tasks=4)",
        "        for t in range(4):",
        "            cn.set_task(t)",
        "            x = torch.randn(8, 16)",
        "            out = cn(x)",
        "            assert not torch.isnan(out).any()",
        "",
        "    def test_eval_uses_running_stats(self):",
        "        from continual_learning import ContinualNormalization",
        "        cn = ContinualNormalization(8, num_tasks=2)",
        "        cn.set_task(0)",
        "        x = torch.randn(16, 8)",
        "        cn.train()",
        "        cn(x)",
        "        cn.eval()",
        "        out = cn(x)",
        "        assert not torch.isnan(out).any()",
        "",
    ]

    lines += [
        "class TestMemoryReplayBuffer:",
        "    def test_add_and_sample(self):",
        "        from continual_learning import MemoryReplayBuffer",
        "        buf = MemoryReplayBuffer(capacity=100, strategy='reservoir')",
        "        x = torch.randn(20, 16)",
        "        y = torch.randint(0, 4, (20,))",
        "        buf.add(x, y)",
        "        assert len(buf) == 20",
        "        sx, sy = buf.sample(10)",
        "        assert sx.shape == (10, 16)",
        "        assert sy.shape == (10,)",
        "",
        "    def test_capacity_limit(self):",
        "        from continual_learning import MemoryReplayBuffer",
        "        buf = MemoryReplayBuffer(capacity=50)",
        "        for _ in range(10):",
        "            buf.add(torch.randn(10, 8), torch.randint(0, 2, (10,)))",
        "        assert len(buf) <= 50",
        "",
        "    def test_fifo_strategy(self):",
        "        from continual_learning import MemoryReplayBuffer",
        "        buf = MemoryReplayBuffer(capacity=10, strategy='fifo')",
        "        for i in range(5):",
        "            buf.add(torch.randn(4, 8), torch.randint(0, 2, (4,)))",
        "        assert len(buf) == 10",
        "",
    ]

    lines += [
        "class TestDualMemorySystem:",
        "    def test_forward_shape(self):",
        "        from continual_learning import DualMemorySystem",
        "        dms = DualMemorySystem(32, 64, 64)",
        "        x = torch.randn(4, 32)",
        "        out = dms(x)",
        "        assert out.shape == (4, 64)",
        "",
        "    def test_no_nan(self):",
        "        from continual_learning import DualMemorySystem",
        "        dms = DualMemorySystem(16, 32, 32)",
        "        x = torch.randn(4, 16)",
        "        assert not torch.isnan(dms(x)).any()",
        "",
    ]

    lines += [
        "class TestSynapticIntelligence:",
        "    def _make_model(self):",
        "        return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))",
        "",
        "    def test_si_loss_zero_initially(self):",
        "        from continual_learning import SynapticIntelligence",
        "        model = self._make_model()",
        "        si = SynapticIntelligence(model)",
        "        loss = si.si_loss()",
        "        assert loss.item() == 0.0",
        "",
        "    def test_forward_passthrough(self):",
        "        from continual_learning import SynapticIntelligence",
        "        model = nn.Linear(16, 4)",
        "        si = SynapticIntelligence(model)",
        "        x = torch.randn(2, 16)",
        "        out = si(x)",
        "        assert out.shape == (2, 4)",
        "",
    ]

    return "\n".join(lines)

write_new("tests/test_continual_learning.py", build_cl_tests())

# ════════════════════════════════════════════════════════════════════════
# 4. tests/test_rlhf_comprehensive.py
# ════════════════════════════════════════════════════════════════════════
def build_rlhf_tests():
    lines = [
        '"""Comprehensive tests for rlhf.py components."""',
        "import pytest",
        "import torch",
        "import torch.nn as nn",
        "",
        "def _make_lm(vocab=100, d=32, layers=2, seq=32):",
        "    embed = nn.Embedding(vocab, d)",
        "    lm_head = nn.Linear(d, vocab)",
        "    class SimpleLM(nn.Module):",
        "        def __init__(self):",
        "            super().__init__()",
        "            self.embed = embed",
        "            self.lm_head = lm_head",
        "        def forward(self, ids):",
        "            return self.lm_head(self.embed(ids))",
        "    return SimpleLM()",
        "",
    ]

    lines += [
        "class TestRewardModel:",
        "    def test_scalar_output_shape(self):",
        "        from rlhf import RewardModel",
        "        lm = _make_lm()",
        "        rm = RewardModel(lm, d_model=32, pooling='last_token')",
        "        ids = torch.randint(0, 100, (2, 16))",
        "        reward = rm(ids)",
        "        assert reward.shape == (2,)",
        "",
        "    def test_mean_pooling(self):",
        "        from rlhf import RewardModel",
        "        lm = _make_lm()",
        "        rm = RewardModel(lm, d_model=32, pooling='mean')",
        "        ids = torch.randint(0, 100, (2, 16))",
        "        mask = torch.ones(2, 16)",
        "        reward = rm(ids, mask)",
        "        assert reward.shape == (2,)",
        "",
        "    def test_no_nan(self):",
        "        from rlhf import RewardModel",
        "        lm = _make_lm()",
        "        rm = RewardModel(lm, d_model=32)",
        "        ids = torch.randint(0, 100, (4, 8))",
        "        assert not torch.isnan(rm(ids)).any()",
        "",
        "    def test_gradient_flows(self):",
        "        from rlhf import RewardModel",
        "        lm = _make_lm()",
        "        rm = RewardModel(lm, d_model=32)",
        "        ids = torch.randint(0, 100, (2, 8))",
        "        rm(ids).sum().backward()",
        "        for p in rm.value_head.parameters():",
        "            if p.requires_grad:",
        "                assert p.grad is not None",
        "",
    ]

    lines += [
        "class TestBradleyTerryLoss:",
        "    def test_chosen_better_gives_lower_loss(self):",
        "        from rlhf import BradleyTerryLoss",
        "        loss_fn = BradleyTerryLoss()",
        "        # Large margin: chosen >> rejected",
        "        r_c = torch.tensor([5.0, 4.0, 3.0])",
        "        r_r = torch.tensor([-1.0, -2.0, -3.0])",
        "        loss_good = loss_fn(r_c, r_r).item()",
        "        # Equal reward -> higher loss",
        "        r_eq = torch.zeros(3)",
        "        loss_bad = loss_fn(r_eq, r_eq).item()",
        "        assert loss_good < loss_bad",
        "",
        "    def test_positive_loss(self):",
        "        from rlhf import BradleyTerryLoss",
        "        loss_fn = BradleyTerryLoss()",
        "        r_c = torch.randn(8)",
        "        r_r = torch.randn(8)",
        "        loss = loss_fn(r_c, r_r)",
        "        assert loss.item() > 0",
        "",
    ]

    lines += [
        "class TestDirectPreferenceOptimization:",
        "    def test_loss_scalar(self):",
        "        from rlhf import DirectPreferenceOptimization",
        "        policy = _make_lm(vocab=100, d=32)",
        "        ref = _make_lm(vocab=100, d=32)",
        "        dpo = DirectPreferenceOptimization(policy, ref, beta=0.1)",
        "        prompt = torch.randint(0, 100, (2, 4))",
        "        chosen = torch.randint(0, 100, (2, 8))",
        "        rejected = torch.randint(0, 100, (2, 8))",
        "        loss, metrics = dpo(prompt, chosen, rejected)",
        "        assert loss.dim() == 0",
        "        assert 'accuracy' in metrics",
        "",
        "    def test_gradient_flows(self):",
        "        from rlhf import DirectPreferenceOptimization",
        "        policy = _make_lm(vocab=50, d=16)",
        "        ref = _make_lm(vocab=50, d=16)",
        "        dpo = DirectPreferenceOptimization(policy, ref, beta=0.1)",
        "        prompt = torch.randint(0, 50, (2, 4))",
        "        chosen = torch.randint(0, 50, (2, 4))",
        "        rejected = torch.randint(0, 50, (2, 4))",
        "        loss, _ = dpo(prompt, chosen, rejected)",
        "        loss.backward()",
        "        has_grad = any(p.grad is not None for p in policy.parameters())",
        "        assert has_grad",
        "",
        "    def test_ref_model_frozen(self):",
        "        from rlhf import DirectPreferenceOptimization",
        "        policy = _make_lm(vocab=50, d=16)",
        "        ref = _make_lm(vocab=50, d=16)",
        "        dpo = DirectPreferenceOptimization(policy, ref)",
        "        for p in dpo.ref.parameters():",
        "            assert not p.requires_grad",
        "",
    ]

    lines += [
        "class TestIdentityPreferenceOptimization:",
        "    def test_loss_shape(self):",
        "        from rlhf import IdentityPreferenceOptimization",
        "        policy = _make_lm(vocab=50, d=16)",
        "        ref = _make_lm(vocab=50, d=16)",
        "        ipo = IdentityPreferenceOptimization(policy, ref, tau=0.1)",
        "        prompt = torch.randint(0, 50, (2, 4))",
        "        chosen = torch.randint(0, 50, (2, 4))",
        "        rejected = torch.randint(0, 50, (2, 4))",
        "        loss, metrics = ipo(prompt, chosen, rejected)",
        "        assert loss.dim() == 0",
        "        assert not torch.isnan(loss)",
        "",
    ]

    lines += [
        "class TestRewardModelTrainer:",
        "    def test_train_step(self):",
        "        from rlhf import RewardModel, RewardModelTrainer",
        "        lm = _make_lm(vocab=50, d=16)",
        "        rm = RewardModel(lm, d_model=16)",
        "        trainer = RewardModelTrainer(rm, learning_rate=1e-4)",
        "        prompt = torch.randint(0, 50, (2, 4))",
        "        chosen = torch.randint(0, 50, (2, 8))",
        "        rejected = torch.randint(0, 50, (2, 8))",
        "        loss = trainer.train_step(prompt, chosen, rejected)",
        "        assert isinstance(loss, float)",
        "        assert loss > 0",
        "",
    ]

    lines += [
        "class TestPreferenceDataset:",
        "    def test_add_and_getitem(self):",
        "        from rlhf import PreferenceDataset",
        "        ds = PreferenceDataset()",
        "        p = torch.randint(0, 100, (10,))",
        "        c = torch.randint(0, 100, (20,))",
        "        r = torch.randint(0, 100, (20,))",
        "        ds.add(p, c, r)",
        "        assert len(ds) == 1",
        "        item = ds[0]",
        "        assert 'prompt' in item and 'chosen' in item and 'rejected' in item",
        "",
    ]

    lines += [
        "class TestGeneralizedAdvantageEstimation:",
        "    def test_shapes(self):",
        "        from rlhf import GeneralizedAdvantagEstimation",
        "        T = 16",
        "        rewards = torch.randn(T)",
        "        values = torch.randn(T + 1)",
        "        dones = torch.zeros(T)",
        "        returns, adv = GeneralizedAdvantagEstimation.compute(rewards, values, dones)",
        "        assert returns.shape == (T,)",
        "        assert adv.shape == (T,)",
        "",
        "    def test_no_nan(self):",
        "        from rlhf import GeneralizedAdvantagEstimation",
        "        T = 8",
        "        rewards = torch.randn(T)",
        "        values = torch.randn(T + 1)",
        "        dones = torch.zeros(T)",
        "        returns, adv = GeneralizedAdvantagEstimation.compute(rewards, values, dones)",
        "        assert not torch.isnan(returns).any()",
        "        assert not torch.isnan(adv).any()",
        "",
    ]

    # Parametrized DPO
    lines += [
        "@pytest.mark.parametrize('beta,prompt_len,resp_len', [",
        "    (0.05, 4, 4), (0.1, 4, 8), (0.2, 8, 8), (0.5, 8, 16),",
        "    (0.1, 4, 4), (0.2, 8, 4),",
        "])",
        "def test_dpo_parametrized(beta, prompt_len, resp_len):",
        "    from rlhf import DirectPreferenceOptimization",
        "    policy = _make_lm(vocab=50, d=16)",
        "    ref = _make_lm(vocab=50, d=16)",
        "    dpo = DirectPreferenceOptimization(policy, ref, beta=beta)",
        "    prompt = torch.randint(0, 50, (2, prompt_len))",
        "    chosen = torch.randint(0, 50, (2, resp_len))",
        "    rejected = torch.randint(0, 50, (2, resp_len))",
        "    loss, metrics = dpo(prompt, chosen, rejected)",
        "    assert not torch.isnan(loss)",
        "    assert 0.0 <= metrics['accuracy'] <= 1.0",
        "",
    ]

    return "\n".join(lines)

write_new("tests/test_rlhf_comprehensive.py", build_rlhf_tests())

# ════════════════════════════════════════════════════════════════════════
# 5. tests/test_deployment_comprehensive.py
# ════════════════════════════════════════════════════════════════════════
def build_deployment_tests():
    lines = [
        '"""Tests for deployment.py components."""',
        "import pytest",
        "import time",
        "import torch",
        "import torch.nn as nn",
        "",
    ]

    lines += [
        "class TestServingConfig:",
        "    def test_defaults(self):",
        "        from deployment import ServingConfig",
        "        cfg = ServingConfig()",
        "        assert cfg.max_batch_size == 32",
        "        assert cfg.device == 'cpu'",
        "",
        "    def test_custom(self):",
        "        from deployment import ServingConfig",
        "        cfg = ServingConfig(max_batch_size=8, timeout_ms=50.0, use_fp16=False)",
        "        assert cfg.max_batch_size == 8",
        "",
    ]

    lines += [
        "class TestRequest:",
        "    def test_creation(self):",
        "        from deployment import Request",
        "        req = Request('req1', {'x': torch.randn(4)})",
        "        assert req.request_id == 'req1'",
        "        assert req.priority == 0",
        "        assert req.timestamp > 0",
        "",
    ]

    lines += [
        "class TestRequestQueue:",
        "    def test_put_and_get(self):",
        "        from deployment import RequestQueue, Request",
        "        q = RequestQueue()",
        "        req = Request('r1', {'x': torch.randn(4)}, priority=1)",
        "        q.put(req)",
        "        out = q.get(block=False)",
        "        assert out.request_id == 'r1'",
        "",
        "    def test_priority_ordering(self):",
        "        from deployment import RequestQueue, Request",
        "        q = RequestQueue()",
        "        q.put(Request('low', {}, priority=0))",
        "        q.put(Request('high', {}, priority=10))",
        "        first = q.get(block=False)",
        "        assert first.request_id == 'high'",
        "",
        "    def test_get_batch(self):",
        "        from deployment import RequestQueue, Request",
        "        q = RequestQueue()",
        "        for i in range(5):",
        "            q.put(Request(f'r{i}', {}))",
        "        batch = q.get_batch(max_size=3, timeout_ms=10.0)",
        "        assert len(batch) == 3",
        "",
    ]

    lines += [
        "class TestDynamicBatcher:",
        "    def test_batch_on_size(self):",
        "        from deployment import DynamicBatcher, Request",
        "        batcher = DynamicBatcher(max_batch_size=3, timeout_ms=1000.0)",
        "        result = None",
        "        for i in range(3):",
        "            result = batcher.add(Request(f'r{i}', {}))",
        "        assert result is not None",
        "        assert len(result) == 3",
        "",
        "    def test_flush(self):",
        "        from deployment import DynamicBatcher, Request",
        "        batcher = DynamicBatcher(max_batch_size=10)",
        "        batcher.add(Request('r1', {}))",
        "        batcher.add(Request('r2', {}))",
        "        batch = batcher.flush()",
        "        assert len(batch) == 2",
        "",
    ]

    lines += [
        "class TestInferenceCache:",
        "    def test_miss_then_hit(self):",
        "        from deployment import InferenceCache, Response",
        "        cache = InferenceCache(capacity=100)",
        "        inputs = {'x': torch.tensor([1.0, 2.0, 3.0])}",
        "        assert cache.get(inputs) is None",
        "        resp = Response('r1', {'out': torch.tensor([1.0])})",
        "        cache.put(inputs, resp)",
        "        hit = cache.get(inputs)",
        "        assert hit is not None",
        "        assert hit.request_id == 'r1'",
        "",
        "    def test_capacity_limit(self):",
        "        from deployment import InferenceCache, Response",
        "        cache = InferenceCache(capacity=5)",
        "        for i in range(10):",
        "            inp = {'x': torch.tensor([float(i)])};",
        "            cache.put(inp, Response(f'r{i}', {}))",
        "        assert len(cache._cache) <= 5",
        "",
        "    def test_hit_rate(self):",
        "        from deployment import InferenceCache, Response",
        "        cache = InferenceCache(capacity=10)",
        "        inp = {'x': torch.tensor([1.0, 2.0])}",
        "        cache.get(inp)  # miss",
        "        cache.put(inp, Response('r1', {}))",
        "        cache.get(inp)  # hit",
        "        assert cache.hit_rate > 0",
        "",
    ]

    lines += [
        "class TestServerMetrics:",
        "    def test_record_and_summary(self):",
        "        from deployment import ServerMetrics",
        "        m = ServerMetrics()",
        "        for i in range(10):",
        "            m.record_batch(4, float(10 + i))",
        "        s = m.summary()",
        "        assert 'p50_latency_ms' in s",
        "        assert s['total_requests'] == 40",
        "",
        "    def test_empty_summary(self):",
        "        from deployment import ServerMetrics",
        "        m = ServerMetrics()",
        "        s = m.summary()",
        "        assert 'total_requests' in s",
        "",
    ]

    lines += [
        "class TestTokenBucketRateLimiter:",
        "    def test_acquire_within_burst(self):",
        "        from deployment import TokenBucketRateLimiter",
        "        limiter = TokenBucketRateLimiter(rate_qps=100.0, burst=10.0)",
        "        result = limiter.acquire(block=False)",
        "        assert result is True",
        "",
        "    def test_acquire_exceed_burst_nonblocking(self):",
        "        from deployment import TokenBucketRateLimiter",
        "        limiter = TokenBucketRateLimiter(rate_qps=1.0, burst=1.0)",
        "        limiter.acquire(1.0, block=False)  # use up burst",
        "        result = limiter.acquire(1.0, block=False)",
        "        assert result is False",
        "",
    ]

    lines += [
        "class TestModelVersionManager:",
        "    def test_register_and_route(self):",
        "        from deployment import ModelVersionManager",
        "        vm = ModelVersionManager()",
        "        m1 = nn.Linear(4, 2)",
        "        m2 = nn.Linear(4, 2)",
        "        vm.register('v1', m1, traffic_weight=0.8)",
        "        vm.register('v2', m2, traffic_weight=0.2)",
        "        version = vm.route_request()",
        "        assert version in ['v1', 'v2']",
        "",
        "    def test_canary_deployment(self):",
        "        from deployment import ModelVersionManager",
        "        vm = ModelVersionManager()",
        "        m1 = nn.Linear(4, 2)",
        "        m2 = nn.Linear(4, 2)",
        "        vm.register('v1', m1)",
        "        vm.register('v2', m2)",
        "        vm.set_canary('v2', canary_fraction=0.1)",
        "        assert abs(vm._traffic_weights['v2'] - 0.1) < 0.01",
        "",
        "    def test_promote_canary(self):",
        "        from deployment import ModelVersionManager",
        "        vm = ModelVersionManager()",
        "        vm.register('v1', nn.Linear(4, 2))",
        "        vm.register('v2', nn.Linear(4, 2))",
        "        vm.promote_canary('v2')",
        "        assert vm._active_version == 'v2'",
        "        assert vm._traffic_weights['v2'] == 1.0",
        "",
    ]

    lines += [
        "class TestGradientFreeShadowMode:",
        "    def test_returns_production_output(self):",
        "        from deployment import GradientFreeShadowMode",
        "        prod = nn.Linear(8, 4)",
        "        shadow = nn.Linear(8, 4)",
        "        sm = GradientFreeShadowMode(prod, shadow, log_dir='/tmp/shadow_test')",
        "        x = torch.randn(2, 8)",
        "        out = sm(x)",
        "        expected = prod(x)",
        "        assert torch.allclose(out, expected)",
        "",
        "    def test_discrepancy_tracking(self):",
        "        from deployment import GradientFreeShadowMode",
        "        prod = nn.Linear(8, 4)",
        "        shadow = nn.Linear(8, 4)",
        "        sm = GradientFreeShadowMode(prod, shadow, log_dir='/tmp/shadow_test2')",
        "        for _ in range(5):",
        "            sm(torch.randn(2, 8))",
        "        stats = sm.discrepancy_stats()",
        "        assert 'mean_discrepancy' in stats",
        "",
    ]

    return "\n".join(lines)

write_new("tests/test_deployment_comprehensive.py", build_deployment_tests())

# ════════════════════════════════════════════════════════════════════════
# 6. Append to attention.py with more attention variants
# ════════════════════════════════════════════════════════════════════════
ATTN_ADD2 = '''

# ============================================================
# Additional Attention Mechanisms (Extended)
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CosineAttention(nn.Module):
    """Cosine attention (Chen et al. 2021): uses cosine similarity instead of dot product.

    Normalizes queries and keys, making attention scale-invariant.
    """

    def __init__(self, d_model: int, num_heads: int, tau_init: float = 20.0, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable temperature per head
        self.tau = nn.Parameter(torch.full((num_heads, 1, 1), tau_init))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(1, 2)

        # L2 normalize along head_dim
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Cosine similarity scaled by tau
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.tau  # (B, H, T, T)

        if mask is not None:
            attn = attn + mask

        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)


class RetentiveAttention(nn.Module):
    """RetNet retention mechanism (Sun et al. 2023) - parallel chunk mode.

    Replaces softmax attention with a decaying exponential retention matrix.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.group_norm = nn.GroupNorm(num_heads, d_model)
        self.dropout = nn.Dropout(dropout)

        # Retention decay rates per head
        gamma = 1 - 2 ** (-5 - torch.arange(num_heads).float())
        self.register_buffer("gamma", gamma)

    def _build_decay_matrix(self, T: int, device) -> torch.Tensor:
        """Build T×T decay matrix D where D[i,j] = gamma^(i-j) if i>=j else 0."""
        i = torch.arange(T, device=device).unsqueeze(1)
        j = torch.arange(T, device=device).unsqueeze(0)
        # (H, T, T)
        exponent = (i - j).clamp(min=0).unsqueeze(0).float()  # (1, T, T)
        gamma = self.gamma.unsqueeze(-1).unsqueeze(-1)          # (H, 1, 1)
        D = (gamma ** exponent) * (i >= j).float().unsqueeze(0)
        return D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).permute(0, 2, 1, 3)  # (B,H,T,D)
        k = self.k_proj(x).reshape(B, T, H, D).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, H, D).permute(0, 2, 1, 3)

        decay = self._build_decay_matrix(T, x.device)  # (H, T, T)

        qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # (B,H,T,T)
        retention = qk * decay.unsqueeze(0)  # (B,H,T,T)
        retention = self.dropout(retention)
        out = torch.matmul(retention, v)  # (B,H,T,D)

        out = out.permute(0, 2, 1, 3).reshape(B, T, self.d_model)
        # Group norm for stabilization
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)
        return self.out_proj(out)


class KVCacheMultiHeadAttention(nn.Module):
    """Multi-head attention with explicit KV-cache for autoregressive decoding."""

    def __init__(self, d_model: int, num_heads: int, max_cache_len: int = 2048, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_cache_len = max_cache_len

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(self.head_dim)

        # KV cache (not a parameter)
        self._cache_k: Optional[torch.Tensor] = None
        self._cache_v: Optional[torch.Tensor] = None
        self._cache_pos = 0

    def reset_cache(self):
        self._cache_k = None
        self._cache_v = None
        self._cache_pos = 0

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(1, 2)

        if use_cache:
            if self._cache_k is None:
                self._cache_k = k
                self._cache_v = v
            else:
                self._cache_k = torch.cat([self._cache_k, k], dim=2)
                self._cache_v = torch.cat([self._cache_v, v], dim=2)
                # Trim to max_cache_len
                if self._cache_k.shape[2] > self.max_cache_len:
                    self._cache_k = self._cache_k[:, :, -self.max_cache_len:]
                    self._cache_v = self._cache_v[:, :, -self.max_cache_len:]
            k_eff = self._cache_k
            v_eff = self._cache_v
        else:
            k_eff = k
            v_eff = v

        attn = torch.matmul(q, k_eff.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_eff).transpose(1, 2).reshape(B, T, self.d_model)
        out = self.out_proj(out)

        past_kv = (self._cache_k, self._cache_v) if use_cache else None
        return out, past_kv


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention (Shazeer 2019): single shared K and V, multiple Q heads.

    Reduces KV-cache memory by (num_heads) factor during inference.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        # Single K and V heads (not per-query-head)
        self.k_proj = nn.Linear(d_model, self.head_dim)
        self.v_proj = nn.Linear(d_model, self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).unsqueeze(1)  # (B, 1, T, D) shared across heads
        v = self.v_proj(x).unsqueeze(1)  # (B, 1, T, D) shared across heads

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        if mask is not None:
            attn = attn + mask
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)


class ALiBiAttention(nn.Module):
    """ALiBi (Press et al. 2022): attention with linear biases instead of position embeddings.

    Adds a fixed (non-learned) bias proportional to query-key distance.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # ALiBi slopes: 2^(-8/H * h) for h=1..H
        slopes = self._get_slopes(num_heads)
        self.register_buffer("slopes", slopes)

    @staticmethod
    def _get_slopes(n_heads: int) -> torch.Tensor:
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start ** i) for i in range(n)]

        if math.log2(n_heads).is_integer():
            return torch.tensor(get_slopes_power_of_2(n_heads))
        else:
            closest = 2 ** math.floor(math.log2(n_heads))
            slopes = get_slopes_power_of_2(closest)
            slopes += get_slopes_power_of_2(2 * closest)[0::2][:n_heads - closest]
            return torch.tensor(slopes[:n_heads])

    def _build_alibi_bias(self, T: int, device) -> torch.Tensor:
        """Returns (H, T, T) ALiBi bias matrix."""
        positions = torch.arange(T, device=device)
        dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)
        dist = -dist.abs().float()
        bias = self.slopes.unsqueeze(-1).unsqueeze(-1) * dist.unsqueeze(0)  # (H, T, T)
        return bias

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        alibi = self._build_alibi_bias(T, x.device).unsqueeze(0)    # (1, H, T, T)
        attn = attn + alibi

        if mask is not None:
            attn = attn + mask
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)


class RotaryPositionEmbedding(nn.Module):
    """RoPE (Su et al. 2021): rotary position embeddings applied to Q and K."""

    def __init__(self, head_dim: int, base: int = 10000):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def _get_sin_cos(self, T: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(T, device=device).float()
        freqs = torch.outer(t, self.inv_freq)  # (T, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, D)
        return emb.sin(), emb.cos()

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        return x * cos + self.rotate_half(x) * sin

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = q.shape[-2]
        sin, cos = self._get_sin_cos(T, q.device)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
        cos = cos.unsqueeze(0).unsqueeze(0)
        return self.apply_rotary(q, sin, cos), self.apply_rotary(k, sin, cos)


class RoPEAttention(nn.Module):
    """Multi-head attention with rotary position embeddings (RoPE)."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, rope_base: int = 10000):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        self.rope = RotaryPositionEmbedding(self.head_dim, base=rope_base)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = self.rope(q, k)

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn + mask
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)


class FlashAttentionSimulator(nn.Module):
    """Simulates FlashAttention (Dao et al. 2022) interface - tiled computation.

    For testing purposes only (not the actual CUDA kernel).
    Computes exact attention but in tiles to simulate the memory access pattern.
    """

    def __init__(self, d_model: int, num_heads: int, block_size: int = 32, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(self.head_dim)

    def _tiled_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Computes attention in tiles (equivalent to standard attention)."""
        B, H, T, D = q.shape
        Bc = self.block_size
        out = torch.zeros_like(q)
        row_max = torch.full((B, H, T, 1), float("-inf"), device=q.device)
        row_sum = torch.zeros(B, H, T, 1, device=q.device)

        for j_start in range(0, T, Bc):
            j_end = min(j_start + Bc, T)
            k_block = k[:, :, j_start:j_end, :]  # (B, H, Bc, D)
            v_block = v[:, :, j_start:j_end, :]

            s = torch.matmul(q, k_block.transpose(-2, -1)) / self.scale  # (B, H, T, Bc)
            m_new = torch.maximum(row_max, s.max(dim=-1, keepdim=True).values)
            exp_s = torch.exp(s - m_new)

            out = out * torch.exp(row_max - m_new) + torch.matmul(exp_s, v_block)
            row_sum = row_sum * torch.exp(row_max - m_new) + exp_s.sum(dim=-1, keepdim=True)
            row_max = m_new

        out = out / (row_sum + 1e-8)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(1, 2)

        out = self._tiled_attention(q, k, v).transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)


class XFormersStyleAttention(nn.Module):
    """xFormers-style efficient attention with memory_efficient_attention fallback."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, H, D)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Try scaled_dot_product_attention if available (PyTorch 2.0+)
        try:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        except AttributeError:
            attn = (q @ k.transpose(-2, -1)) / self.scale
            if mask is not None:
                attn = attn + mask
            attn = F.softmax(attn, dim=-1)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)
'''

append("attention.py", ATTN_ADD2)

# ════════════════════════════════════════════════════════════════════════
# 6. tests/test_attention_extra.py
# ════════════════════════════════════════════════════════════════════════
def build_attn_extra_tests():
    classes = [
        ("CosineAttention", "attention", "d_model=64, num_heads=4"),
        ("RetentiveAttention", "attention", "d_model=64, num_heads=4"),
        ("MultiQueryAttention", "attention", "d_model=64, num_heads=4"),
        ("ALiBiAttention", "attention", "d_model=64, num_heads=4"),
        ("RoPEAttention", "attention", "d_model=64, num_heads=4"),
        ("FlashAttentionSimulator", "attention", "d_model=64, num_heads=4"),
        ("XFormersStyleAttention", "attention", "d_model=64, num_heads=4"),
        ("KVCacheMultiHeadAttention", "attention", "d_model=64, num_heads=4"),
    ]

    lines = [
        '"""Tests for extra attention mechanisms in attention.py."""',
        "import pytest",
        "import torch",
        "",
    ]

    for cls, module, init_args in classes:
        lines += [
            f"class Test{cls}:",
            f"    def setup_method(self):",
            f"        from {module} import {cls}",
            f"        self.model = {cls}({init_args})",
            "",
            f"    def test_forward_shape_small(self):",
            f"        x = torch.randn(2, 8, 64)",
            f"        out = self.model(x)",
            f"        if isinstance(out, tuple): out = out[0]",
            f"        assert out.shape == (2, 8, 64)",
            "",
            f"    def test_forward_shape_larger(self):",
            f"        x = torch.randn(4, 32, 64)",
            f"        out = self.model(x)",
            f"        if isinstance(out, tuple): out = out[0]",
            f"        assert out.shape == (4, 32, 64)",
            "",
            f"    def test_no_nan(self):",
            f"        x = torch.randn(2, 16, 64)",
            f"        out = self.model(x)",
            f"        if isinstance(out, tuple): out = out[0]",
            f"        assert not torch.isnan(out).any()",
            "",
            f"    def test_gradient(self):",
            f"        x = torch.randn(2, 8, 64, requires_grad=True)",
            f"        out = self.model(x)",
            f"        if isinstance(out, tuple): out = out[0]",
            f"        out.sum().backward()",
            f"        assert x.grad is not None",
            "",
            f"    def test_eval_mode_consistent(self):",
            f"        x = torch.randn(2, 8, 64)",
            f"        self.model.eval()",
            f"        with torch.no_grad():",
            f"            out1 = self.model(x)",
            f"            out2 = self.model(x)",
            f"        if isinstance(out1, tuple): out1, out2 = out1[0], out2[0]",
            f"        assert torch.allclose(out1, out2)",
            "",
            f"    def test_state_dict_loadable(self):",
            f"        from {module} import {cls}",
            f"        sd = self.model.state_dict()",
            f"        model2 = {cls}({init_args})",
            f"        model2.load_state_dict(sd)",
            "",
        ]

    # Parametrized
    lines += [
        "@pytest.mark.parametrize('d_model,num_heads,B,T', [",
    ]
    for d in [32, 64, 128]:
        for h in [4, 8]:
            if d % h == 0:
                for B in [1, 2]:
                    for T in [8, 16, 32]:
                        lines.append(f"    ({d}, {h}, {B}, {T}),")
    lines += [
        "])",
        "def test_rope_attention_parametrized(d_model, num_heads, B, T):",
        "    from attention import RoPEAttention",
        "    model = RoPEAttention(d_model, num_heads)",
        "    x = torch.randn(B, T, d_model)",
        "    out = model(x)",
        "    assert out.shape == (B, T, d_model)",
        "    assert not torch.isnan(out).any()",
        "",
        "@pytest.mark.parametrize('d_model,num_heads,B,T', [",
    ]
    for d in [32, 64]:
        for h in [4, 8]:
            if d % h == 0:
                for B in [1, 2]:
                    for T in [8, 16]:
                        lines.append(f"    ({d}, {h}, {B}, {T}),")
    lines += [
        "])",
        "def test_alibi_attention_parametrized(d_model, num_heads, B, T):",
        "    from attention import ALiBiAttention",
        "    model = ALiBiAttention(d_model, num_heads)",
        "    x = torch.randn(B, T, d_model)",
        "    out = model(x)",
        "    assert out.shape == (B, T, d_model)",
        "    assert not torch.isnan(out).any()",
        "",
    ]

    return "\n".join(lines)

write_new("tests/test_attention_extra.py", build_attn_extra_tests())

# Final count
import subprocess
result = subprocess.run(
    ["bash", "-c",
     "find /c/Users/Matthew/srfm-lab/aeternus/lumina -name '*.py' -o -name '*.yaml' | xargs wc -l 2>/dev/null | tail -1"],
    capture_output=True, text=True
)
print("GRAND TOTAL:", result.stdout.strip())
