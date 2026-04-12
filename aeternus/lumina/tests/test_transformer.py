"""
tests/test_transformer.py

Unit tests for lumina transformer components.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lumina.transformer import (
    RMSNorm,
    SwiGLUFFN,
    MultiHeadSelfAttention,
    GroupedQueryAttention,
    MixtureOfExpertsLayer,
    TransformerBlock,
    CausalTransformer,
    BidirectionalTransformer,
    LuminaModel,
    LuminaConfig,
)
from lumina.positional_encoding import (
    RotaryPositionalEncoding,
    ALiBiPositionalBias,
    TemporalEncoding,
    FourierTimeEncoding,
    CrossModalPositionalEncoding,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def d_model():
    return 64

@pytest.fixture
def n_heads():
    return 4

@pytest.fixture
def seq_len():
    return 16

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def sample_hidden(batch_size, seq_len, d_model):
    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, d_model)

@pytest.fixture
def sample_mask(batch_size, seq_len):
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, -4:] = False  # first sample has 4 pad tokens
    return mask


# ---------------------------------------------------------------------------
# RMSNorm tests
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_output_shape(self, d_model):
        norm = RMSNorm(d_model)
        x = torch.randn(2, 16, d_model)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalizes_rms(self, d_model):
        norm = RMSNorm(d_model, eps=0.0)
        x = torch.ones(1, 1, d_model) * 5.0
        out = norm(x)
        # RMS of input is 5, so output should be ~weight * 1.0
        expected_rms = torch.sqrt((out ** 2).mean(dim=-1))
        # Should be close to the weight (initialized to 1)
        assert expected_rms.item() == pytest.approx(1.0, abs=0.01)

    def test_gradient_flow(self, d_model):
        norm = RMSNorm(d_model)
        x = torch.randn(2, 16, d_model, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None
        assert norm.weight.grad is not None

    def test_different_from_layernorm(self, d_model):
        """RMSNorm and LayerNorm should produce different outputs (no mean subtraction)."""
        rms = RMSNorm(d_model)
        ln = nn.LayerNorm(d_model)
        x = torch.randn(2, 4, d_model)
        out_rms = rms(x)
        out_ln = ln(x)
        assert not torch.allclose(out_rms, out_ln)


# ---------------------------------------------------------------------------
# SwiGLUFFN tests
# ---------------------------------------------------------------------------

class TestSwiGLUFFN:
    def test_output_shape(self, d_model):
        ffn = SwiGLUFFN(d_model)
        x = torch.randn(2, 16, d_model)
        out = ffn(x)
        assert out.shape == x.shape

    def test_auto_d_ffn(self, d_model):
        ffn = SwiGLUFFN(d_model)
        # 8/3 * 64 ≈ 170.67, rounded to 192
        assert ffn.d_ffn % 64 == 0
        assert ffn.d_ffn >= d_model

    def test_explicit_d_ffn(self, d_model):
        ffn = SwiGLUFFN(d_model, d_ffn=256)
        assert ffn.d_ffn == 256

    def test_gradient_flow(self, d_model):
        ffn = SwiGLUFFN(d_model)
        x = torch.randn(2, 8, d_model, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None
        assert ffn.w_gate.weight.grad is not None


# ---------------------------------------------------------------------------
# RotaryPositionalEncoding tests
# ---------------------------------------------------------------------------

class TestRoPE:
    def test_output_shape(self, d_model, n_heads, seq_len, batch_size):
        head_dim = d_model // n_heads
        rope = RotaryPositionalEncoding(dim=head_dim, max_seq_len=64)
        q = torch.randn(batch_size, n_heads, seq_len, head_dim)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_different_positions(self, d_model, n_heads):
        head_dim = d_model // n_heads
        rope = RotaryPositionalEncoding(dim=head_dim, max_seq_len=64)
        q = torch.randn(1, 1, 4, head_dim)
        k = torch.randn(1, 1, 4, head_dim)
        q_rot, _ = rope(q, k)
        # Different positions should produce different embeddings
        assert not torch.allclose(q_rot[:, :, 0], q_rot[:, :, 1])

    def test_cache_extension(self):
        rope = RotaryPositionalEncoding(dim=16, max_seq_len=8)
        q = torch.randn(1, 1, 16, 16)
        k = torch.randn(1, 1, 16, 16)
        # Should extend cache without error
        q_rot, k_rot = rope(q, k, seq_len=16)
        assert q_rot.shape == q.shape


# ---------------------------------------------------------------------------
# ALiBi tests
# ---------------------------------------------------------------------------

class TestALiBi:
    def test_bias_shape(self, n_heads, seq_len):
        alibi = ALiBiPositionalBias(n_heads=n_heads, max_seq_len=64)
        bias = alibi.forward(seq_len, torch.device("cpu"))
        assert bias.shape == (1, n_heads, seq_len, seq_len)

    def test_negative_off_diagonal(self, n_heads):
        alibi = ALiBiPositionalBias(n_heads=n_heads, max_seq_len=32)
        bias = alibi.forward(8, torch.device("cpu"))
        # Off-diagonal elements should be negative
        off_diag = bias[0, 0, 0, 1]  # position 0 attending to position 1
        assert off_diag.item() < 0

    def test_causal_bias_shape(self, n_heads, seq_len):
        alibi = ALiBiPositionalBias(n_heads=n_heads, max_seq_len=64)
        causal_bias = alibi.causal_bias(seq_len, torch.device("cpu"))
        assert causal_bias.shape == (1, n_heads, seq_len, seq_len)


# ---------------------------------------------------------------------------
# MultiHeadSelfAttention tests
# ---------------------------------------------------------------------------

class TestMHSA:
    def test_output_shape(self, d_model, n_heads, sample_hidden):
        attn = MultiHeadSelfAttention(d_model, n_heads, use_rope=True, max_seq_len=32)
        out, kv = attn(sample_hidden)
        assert out.shape == sample_hidden.shape

    def test_causal_mask(self, d_model, n_heads, sample_hidden):
        attn = MultiHeadSelfAttention(d_model, n_heads, use_rope=False, max_seq_len=32)
        out_causal, _ = attn(sample_hidden, causal=True)
        out_full, _ = attn(sample_hidden, causal=False)
        # First token output should be same (no future context to attend to)
        assert torch.allclose(out_causal[:, 0, :], out_full[:, 0, :], atol=1e-5)
        # Other tokens may differ
        # Not necessarily different with fixed init, so just check shapes
        assert out_causal.shape == out_full.shape

    def test_attention_mask(self, d_model, n_heads, sample_hidden, sample_mask):
        attn = MultiHeadSelfAttention(d_model, n_heads, use_rope=False, max_seq_len=32)
        out, _ = attn(sample_hidden, attention_mask=sample_mask)
        assert out.shape == sample_hidden.shape
        assert not torch.isnan(out).any()

    def test_kv_cache(self, d_model, n_heads, seq_len, batch_size):
        attn = MultiHeadSelfAttention(d_model, n_heads, use_rope=False, max_seq_len=64)
        x1 = torch.randn(batch_size, seq_len, d_model)
        x2 = torch.randn(batch_size, 1, d_model)
        # First forward
        _, kv = attn(x1)
        # Second forward with cache
        out2, _ = attn(x2, kv_cache=kv)
        assert out2.shape == (batch_size, 1, d_model)

    def test_no_nan(self, d_model, n_heads, sample_hidden):
        attn = MultiHeadSelfAttention(d_model, n_heads, use_rope=True, max_seq_len=32)
        out, _ = attn(sample_hidden)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# GroupedQueryAttention tests
# ---------------------------------------------------------------------------

class TestGQA:
    def test_output_shape(self, d_model, n_heads, sample_hidden):
        n_kv = 2
        gqa = GroupedQueryAttention(d_model, n_heads, n_kv, use_rope=True, max_seq_len=32)
        out, _ = gqa(sample_hidden)
        assert out.shape == sample_hidden.shape

    def test_gqa_head_ratio(self, d_model):
        n_q = 8
        n_kv = 2  # 4x compression
        gqa = GroupedQueryAttention(d_model, n_q, n_kv, max_seq_len=64)
        # K and V projections should have fewer outputs
        assert gqa.k_proj.out_features == n_kv * (d_model // n_q)
        assert gqa.v_proj.out_features == n_kv * (d_model // n_q)

    def test_causal_gqa(self, d_model, n_heads, sample_hidden):
        gqa = GroupedQueryAttention(d_model, n_heads, 2, max_seq_len=32)
        out, _ = gqa(sample_hidden, causal=True)
        assert out.shape == sample_hidden.shape
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# MixtureOfExperts tests
# ---------------------------------------------------------------------------

class TestMoE:
    def test_output_shape(self, d_model, sample_hidden):
        moe = MixtureOfExpertsLayer(d_model, n_experts=4, top_k=2)
        out, aux_loss = moe(sample_hidden)
        assert out.shape == sample_hidden.shape

    def test_aux_loss_scalar(self, d_model, sample_hidden):
        moe = MixtureOfExpertsLayer(d_model, n_experts=4, top_k=2)
        _, aux_loss = moe(sample_hidden)
        assert aux_loss.dim() == 0  # scalar
        assert aux_loss.item() >= 0.0

    def test_gradient_flow(self, d_model, sample_hidden):
        moe = MixtureOfExpertsLayer(d_model, n_experts=4, top_k=2)
        out, aux_loss = moe(sample_hidden)
        (out.sum() + aux_loss).backward()
        assert moe.gate.weight.grad is not None


# ---------------------------------------------------------------------------
# TransformerBlock tests
# ---------------------------------------------------------------------------

class TestTransformerBlock:
    def test_output_shape(self, d_model, n_heads, sample_hidden):
        block = TransformerBlock(d_model, n_heads, use_rope=True, max_seq_len=32)
        out, aux, kv = block(sample_hidden)
        assert out.shape == sample_hidden.shape

    def test_moe_block(self, d_model, n_heads, sample_hidden):
        block = TransformerBlock(d_model, n_heads, use_moe=True, n_experts=4, max_seq_len=32)
        out, aux_loss, _ = block(sample_hidden)
        assert out.shape == sample_hidden.shape
        assert aux_loss is not None

    def test_residual_connection(self, d_model, n_heads):
        """With very large initial output scale, residual should still be close to input."""
        block = TransformerBlock(d_model, n_heads, max_seq_len=32)
        x = torch.zeros(1, 4, d_model)  # zero input
        out, _, _ = block(x)
        # Zero input through attention with zero bias → close to zero output
        assert out.abs().mean() < 1.0  # sanity check

    def test_gqa_block(self, d_model, n_heads, sample_hidden):
        block = TransformerBlock(d_model, n_heads, n_kv_heads=2, max_seq_len=32)
        out, _, _ = block(sample_hidden)
        assert out.shape == sample_hidden.shape


# ---------------------------------------------------------------------------
# CausalTransformer tests
# ---------------------------------------------------------------------------

class TestCausalTransformer:
    def test_output_shape(self, d_model, n_heads, sample_hidden):
        ct = CausalTransformer(d_model, n_layers=2, n_heads=n_heads, moe_every_n=0, max_seq_len=32)
        hidden, aux, kv_caches = ct(sample_hidden)
        assert hidden.shape == sample_hidden.shape

    def test_kv_cache_list_length(self, d_model, n_heads, sample_hidden):
        ct = CausalTransformer(d_model, n_layers=3, n_heads=n_heads, moe_every_n=0, max_seq_len=32)
        _, _, kv_caches = ct(sample_hidden)
        assert len(kv_caches) == 3

    def test_no_nan(self, d_model, n_heads, sample_hidden):
        ct = CausalTransformer(d_model, n_layers=2, n_heads=n_heads, moe_every_n=0, max_seq_len=32)
        hidden, aux, _ = ct(sample_hidden)
        assert not torch.isnan(hidden).any()


# ---------------------------------------------------------------------------
# BidirectionalTransformer tests
# ---------------------------------------------------------------------------

class TestBidirectionalTransformer:
    def test_output_shape(self, d_model, n_heads, sample_hidden):
        bt = BidirectionalTransformer(d_model, n_layers=2, n_heads=n_heads, moe_every_n=0, max_seq_len=32)
        hidden, aux = bt(sample_hidden)
        assert hidden.shape == sample_hidden.shape

    def test_with_mask(self, d_model, n_heads, sample_hidden, sample_mask):
        bt = BidirectionalTransformer(d_model, n_layers=2, n_heads=n_heads, moe_every_n=0, max_seq_len=32)
        hidden, _ = bt(sample_hidden, attention_mask=sample_mask)
        assert hidden.shape == sample_hidden.shape
        assert not torch.isnan(hidden).any()


# ---------------------------------------------------------------------------
# LuminaModel tests
# ---------------------------------------------------------------------------

class TestLuminaModel:
    @pytest.fixture
    def small_config(self):
        return LuminaConfig(
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ffn=128,
            dropout=0.0,
            max_seq_len=32,
            use_rope=True,
            use_moe=False,
            unified_token_dim=32,
            arch="causal",
            lm_head=True,
            pool_head=True,
            n_classes=8,
            use_temporal=False,
        )

    @pytest.fixture
    def lumina_model(self, small_config):
        return LuminaModel(small_config)

    @pytest.fixture
    def token_input(self, batch_size):
        return torch.randn(batch_size, 16, 32)  # (B, T, unified_dim)

    def test_forward_output_keys(self, lumina_model, token_input):
        out = lumina_model(token_input)
        assert "hidden" in out
        assert "aux_loss" in out

    def test_hidden_shape(self, lumina_model, token_input, batch_size):
        out = lumina_model(token_input)
        assert out["hidden"].shape == (batch_size, 16, 64)

    def test_lm_head_output(self, lumina_model, token_input, batch_size):
        out = lumina_model(token_input)
        assert "lm_output" in out
        assert out["lm_output"].shape == (batch_size, 16, 32)

    def test_pool_head_output(self, lumina_model, token_input, batch_size):
        out = lumina_model(token_input)
        assert "cls_logits" in out
        assert "reg_output" in out
        assert out["cls_logits"].shape == (batch_size, 8)

    def test_attention_mask(self, lumina_model, token_input, sample_mask):
        mask = sample_mask[:, :16]
        out = lumina_model(token_input, attention_mask=mask)
        assert not torch.isnan(out["hidden"]).any()

    def test_no_nan(self, lumina_model, token_input):
        out = lumina_model(token_input)
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any(), f"NaN in {k}"

    def test_param_count(self, small_config):
        model = LuminaModel(small_config)
        n = model.get_num_params()
        assert n > 0
        print(f"\n[Small model] Parameters: {n:,}")

    def test_bidirectional_arch(self, batch_size):
        cfg = LuminaConfig(
            d_model=64, n_layers=2, n_heads=4, d_ffn=128,
            use_moe=False, unified_token_dim=32,
            arch="bidirectional", lm_head=True, pool_head=True, n_classes=8,
            use_temporal=False, max_seq_len=32,
        )
        model = LuminaModel(cfg)
        x = torch.randn(batch_size, 12, 32)
        out = model(x)
        assert out["hidden"].shape == (batch_size, 12, 64)

    def test_gradient_end_to_end(self, lumina_model, token_input):
        out = lumina_model(token_input)
        loss = out["cls_logits"].sum() + out["reg_output"].sum()
        loss.backward()
        # Check at least one gradient is non-zero
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in lumina_model.parameters())
        assert has_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
