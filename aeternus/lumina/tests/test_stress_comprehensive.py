"""Stress tests and edge case tests for Lumina components."""
import pytest
import torch
import torch.nn as nn
import math

# ═══ Memory and batch stress ══════════════════════════════════════════════

class TestStressCosineAttention:
    @pytest.mark.parametrize('B,T', [(1, 128), (2, 64), (4, 32), (8, 16)])
    def test_various_batch_seq(self, B, T):
        from attention import CosineAttention
        model = CosineAttention(d_model=64, num_heads=4)
        x = torch.randn(B, T, 64)
        with torch.no_grad():
            out = model(x)
            if isinstance(out, tuple): out = out[0]
        assert out.shape == (B, T, 64)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize('d_model,num_heads', [(32, 4), (64, 8), (128, 8), (256, 8)])
    def test_various_sizes(self, d_model, num_heads):
        if d_model % num_heads != 0:
            pytest.skip('incompatible dims')
        from attention import CosineAttention
        model = CosineAttention(d_model=d_model, num_heads=num_heads)
        x = torch.randn(2, 8, d_model)
        with torch.no_grad():
            out = model(x)
            if isinstance(out, tuple): out = out[0]
        assert out.shape == (2, 8, d_model)

class TestStressRetentiveAttention:
    @pytest.mark.parametrize('B,T', [(1, 128), (2, 64), (4, 32), (8, 16)])
    def test_various_batch_seq(self, B, T):
        from attention import RetentiveAttention
        model = RetentiveAttention(d_model=64, num_heads=4)
        x = torch.randn(B, T, 64)
        with torch.no_grad():
            out = model(x)
            if isinstance(out, tuple): out = out[0]
        assert out.shape == (B, T, 64)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize('d_model,num_heads', [(32, 4), (64, 8), (128, 8), (256, 8)])
    def test_various_sizes(self, d_model, num_heads):
        if d_model % num_heads != 0:
            pytest.skip('incompatible dims')
        from attention import RetentiveAttention
        model = RetentiveAttention(d_model=d_model, num_heads=num_heads)
        x = torch.randn(2, 8, d_model)
        with torch.no_grad():
            out = model(x)
            if isinstance(out, tuple): out = out[0]
        assert out.shape == (2, 8, d_model)

class TestStressMultiQueryAttention:
    @pytest.mark.parametrize('B,T', [(1, 128), (2, 64), (4, 32), (8, 16)])
    def test_various_batch_seq(self, B, T):
        from attention import MultiQueryAttention
        model = MultiQueryAttention(d_model=64, num_heads=4)
        x = torch.randn(B, T, 64)
        with torch.no_grad():
            out = model(x)
            if isinstance(out, tuple): out = out[0]
        assert out.shape == (B, T, 64)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize('d_model,num_heads', [(32, 4), (64, 8), (128, 8), (256, 8)])
    def test_various_sizes(self, d_model, num_heads):
        if d_model % num_heads != 0:
            pytest.skip('incompatible dims')
        from attention import MultiQueryAttention
        model = MultiQueryAttention(d_model=d_model, num_heads=num_heads)
        x = torch.randn(2, 8, d_model)
        with torch.no_grad():
            out = model(x)
            if isinstance(out, tuple): out = out[0]
        assert out.shape == (2, 8, d_model)

class TestStressALiBiAttention:
    @pytest.mark.parametrize('B,T', [(1, 128), (2, 64), (4, 32), (8, 16)])
    def test_various_batch_seq(self, B, T):
        from attention import ALiBiAttention
        model = ALiBiAttention(d_model=64, num_heads=4)
        x = torch.randn(B, T, 64)
        with torch.no_grad():
            out = model(x)
            if isinstance(out, tuple): out = out[0]
        assert out.shape == (B, T, 64)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize('d_model,num_heads', [(32, 4), (64, 8), (128, 8), (256, 8)])
    def test_various_sizes(self, d_model, num_heads):
        if d_model % num_heads != 0:
            pytest.skip('incompatible dims')
        from attention import ALiBiAttention
        model = ALiBiAttention(d_model=d_model, num_heads=num_heads)
        x = torch.randn(2, 8, d_model)
        with torch.no_grad():
            out = model(x)
            if isinstance(out, tuple): out = out[0]
        assert out.shape == (2, 8, d_model)

class TestStressRoPEAttention:
    @pytest.mark.parametrize('B,T', [(1, 128), (2, 64), (4, 32), (8, 16)])
    def test_various_batch_seq(self, B, T):
        from attention import RoPEAttention
        model = RoPEAttention(d_model=64, num_heads=4)
        x = torch.randn(B, T, 64)
        with torch.no_grad():
            out = model(x)
            if isinstance(out, tuple): out = out[0]
        assert out.shape == (B, T, 64)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize('d_model,num_heads', [(32, 4), (64, 8), (128, 8), (256, 8)])
    def test_various_sizes(self, d_model, num_heads):
        if d_model % num_heads != 0:
            pytest.skip('incompatible dims')
        from attention import RoPEAttention
        model = RoPEAttention(d_model=d_model, num_heads=num_heads)
        x = torch.randn(2, 8, d_model)
        with torch.no_grad():
            out = model(x)
            if isinstance(out, tuple): out = out[0]
        assert out.shape == (2, 8, d_model)

# ═══ Edge case tests ═════════════════════════════════════════════════════

class TestEdgeCases:
    def test_single_token_attention(self):
        from attention import RoPEAttention
        model = RoPEAttention(64, 4)
        x = torch.randn(2, 1, 64)
        out = model(x)
        assert out.shape == (2, 1, 64)

    def test_batch_size_one(self):
        from attention import ALiBiAttention
        model = ALiBiAttention(64, 4)
        x = torch.randn(1, 16, 64)
        out = model(x)
        assert out.shape == (1, 16, 64)

    def test_zero_input(self):
        from attention import CosineAttention
        model = CosineAttention(32, 4)
        x = torch.zeros(2, 8, 32)
        out = model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_large_values(self):
        from attention import MultiQueryAttention
        model = MultiQueryAttention(32, 4)
        x = torch.randn(2, 8, 32) * 10
        out = model(x)
        assert not torch.isnan(out).any()

    def test_gradient_clipping_safe(self):
        from lora import LoRALinear
        layer = LoRALinear(32, 64, rank=4)
        x = torch.randn(2, 8, 32)
        out = layer(x).sum()
        out.backward()
        nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
        for p in layer.parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any()

    def test_transformer_single_layer(self):
        from transformer import NormFormerBlock
        block = NormFormerBlock(32, 4)
        x = torch.randn(2, 4, 32)
        out = block(x)
        assert out.shape == (2, 4, 32)

    def test_moe_all_same_expert(self):
        from moe import SparseMoELayer
        moe = SparseMoELayer(32, num_experts=4, top_k=1)
        x = torch.zeros(2, 4, 32)  # identical tokens -> same routing
        out = moe(x)
        assert not torch.isnan(out).any()

    def test_lora_zero_rank(self):
        # rank=1 minimal case
        from lora import LoRALinear
        layer = LoRALinear(16, 32, rank=1, alpha=2.0)
        x = torch.randn(2, 4, 16)
        out = layer(x)
        assert out.shape == (2, 4, 32)

    def test_seq_len_equals_one_transformer(self):
        from transformer import SandwichTransformerBlock
        block = SandwichTransformerBlock(64, 4)
        x = torch.randn(4, 1, 64)
        out = block(x)
        assert out.shape == (4, 1, 64)

# ═══ Numerical stability ════════════════════════════════════════════════

class TestNumericalStability:
    def test_attention_no_overflow_fp32(self):
        from attention import RoPEAttention
        model = RoPEAttention(64, 4)
        x = torch.randn(2, 16, 64) * 100
        out = model(x)
        assert not torch.isinf(out).any()

    def test_lora_with_large_rank(self):
        from lora import LoRALinear
        layer = LoRALinear(64, 128, rank=32, alpha=64.0)
        x = torch.randn(2, 8, 64)
        out = layer(x)
        assert not torch.isnan(out).any()

    def test_moe_router_no_nan(self):
        from moe import TopKRouter
        router = TopKRouter(64, 16, top_k=4)
        x = torch.randn(64, 64)
        out = router(x)
        assert not torch.isnan(out.router_probs).any()

    def test_conformer_with_bn(self):
        from transformer import ConformerBlock
        block = ConformerBlock(64, 4, conv_kernel=31)
        block.train()
        x = torch.randn(8, 64, 64)  # larger batch for BN
        out = block(x)
        assert not torch.isnan(out).any()

    def test_continual_norm_no_div_zero(self):
        from continual_learning import ContinualNormalization
        cn = ContinualNormalization(32, num_tasks=2)
        cn.set_task(0)
        x = torch.ones(8, 32)  # constant input -> var=0
        cn.train()
        out = cn(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_dpo_symmetric_gives_zero_reward_margin(self):
        from rlhf import DirectPreferenceOptimization
        import torch.nn as nn
        # When policy==reference, reward margin should be ~0
        class PassThrough(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(50, 16)
                self.head = nn.Linear(16, 50, bias=False)
            def forward(self, ids): return self.head(self.emb(ids))
        m = PassThrough()
        dpo = DirectPreferenceOptimization(m, m, beta=0.1)
        prompt = torch.randint(0, 50, (2, 4))
        chosen = torch.randint(0, 50, (2, 4))
        rejected = torch.randint(0, 50, (2, 4))
        loss, metrics = dpo(prompt, chosen, rejected)
        assert abs(metrics['reward_margin']) < 1e-3

# ═══ Throughput benchmarks ═══════════════════════════════════════════════

class TestThroughput:
    def test_attention_forward_backward_perf(self):
        import time
        from attention import RoPEAttention
        model = RoPEAttention(64, 4)
        x = torch.randn(4, 64, 64)
        # Warm up
        for _ in range(3):
            model(x).sum().backward()
        # Time
        t0 = time.time()
        for _ in range(20):
            out = model(x)
            out.sum().backward()
        elapsed = time.time() - t0
        print(f'RoPEAttention 20 fwd+bwd: {elapsed:.3f}s')
        assert elapsed < 60.0  # should be fast

    def test_lora_vs_dense_overhead(self):
        import time
        from lora import LoRALinear
        dense = nn.Linear(256, 512)
        lora = LoRALinear(256, 512, rank=16)
        x = torch.randn(32, 64, 256)
        # Warm up
        for _ in range(3):
            dense(x)
            lora(x)
        t0 = time.time()
        for _ in range(50):
            dense(x)
        t_dense = time.time() - t0
        t0 = time.time()
        for _ in range(50):
            lora(x)
        t_lora = time.time() - t0
        # LoRA should not be more than 3x slower than dense
        assert t_lora < 3 * t_dense + 1.0
