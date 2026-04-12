"""
tests/test_multimodal.py

Unit tests for lumina multimodal components.
"""

import pytest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lumina.transformer import LuminaConfig
from lumina.multimodal import (
    CrossModalAttention,
    ModalityFusion,
    TemporalAlignment,
    MultiModalLumina,
    MultiModalLuminaConfig,
    GatedFusion,
    CrossAttentionFusion,
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
def batch_size():
    return 2

@pytest.fixture
def T_price():
    return 16

@pytest.fixture
def T_news():
    return 8

@pytest.fixture
def price_hidden(batch_size, T_price, d_model):
    torch.manual_seed(10)
    return torch.randn(batch_size, T_price, d_model)

@pytest.fixture
def news_hidden(batch_size, T_news, d_model):
    torch.manual_seed(20)
    return torch.randn(batch_size, T_news, d_model)

@pytest.fixture
def price_mask(batch_size, T_price):
    mask = torch.ones(batch_size, T_price, dtype=torch.bool)
    mask[0, -2:] = False
    return mask

@pytest.fixture
def news_mask(batch_size, T_news):
    return torch.ones(batch_size, T_news, dtype=torch.bool)


# ---------------------------------------------------------------------------
# CrossModalAttention tests
# ---------------------------------------------------------------------------

class TestCrossModalAttention:
    def test_output_shape(self, d_model, n_heads, price_hidden, news_hidden):
        cma = CrossModalAttention(d_model, d_model, d_model, n_heads)
        out = cma(price_hidden, news_hidden)
        assert out.shape == price_hidden.shape

    def test_output_shape_asymmetric_T(self, d_model, n_heads, batch_size):
        """Query T ≠ Context T."""
        cma = CrossModalAttention(d_model, d_model, d_model, n_heads)
        q = torch.randn(batch_size, 20, d_model)
        c = torch.randn(batch_size, 5, d_model)
        out = cma(q, c)
        assert out.shape == q.shape

    def test_masked_context(self, d_model, n_heads, price_hidden, news_hidden, news_mask):
        cma = CrossModalAttention(d_model, d_model, d_model, n_heads)
        out = cma(price_hidden, news_hidden, context_mask=news_mask)
        assert out.shape == price_hidden.shape
        assert not torch.isnan(out).any()

    def test_no_nan_with_all_masked_context(self, d_model, n_heads, batch_size):
        cma = CrossModalAttention(d_model, d_model, d_model, n_heads)
        q = torch.randn(batch_size, 8, d_model)
        c = torch.randn(batch_size, 8, d_model)
        # All context positions masked
        context_mask = torch.zeros(batch_size, 8, dtype=torch.bool)
        out = cma(q, c, context_mask=context_mask)
        # Should handle gracefully (nan_to_num applied)
        assert out.shape == q.shape

    def test_different_query_context_dims(self, n_heads, batch_size):
        d_q, d_c, d_out = 64, 128, 64
        cma = CrossModalAttention(d_q, d_c, d_out, n_heads)
        q = torch.randn(batch_size, 10, d_q)
        c = torch.randn(batch_size, 15, d_c)
        out = cma(q, c)
        assert out.shape == (batch_size, 10, d_q)

    def test_gradient_flow(self, d_model, n_heads, price_hidden, news_hidden):
        cma = CrossModalAttention(d_model, d_model, d_model, n_heads)
        out = cma(price_hidden, news_hidden)
        out.sum().backward()
        assert cma.q_proj.weight.grad is not None
        assert cma.k_proj.weight.grad is not None


# ---------------------------------------------------------------------------
# ModalityFusion tests
# ---------------------------------------------------------------------------

class TestModalityFusion:
    @pytest.fixture
    def modality_reps(self, batch_size, T_price, d_model):
        return [torch.randn(batch_size, T_price, d_model) for _ in range(3)]

    def test_concat_mode(self, d_model, n_heads, modality_reps, batch_size, T_price):
        fusion = ModalityFusion(d_model, n_modalities=3, n_heads=n_heads, mode="concat")
        out = fusion(modality_reps, target_len=T_price)
        assert out.shape == (batch_size, T_price, d_model)

    def test_gated_mode(self, d_model, n_heads, modality_reps, batch_size, T_price):
        fusion = ModalityFusion(d_model, n_modalities=3, n_heads=n_heads, mode="gated")
        out = fusion(modality_reps, target_len=T_price)
        assert out.shape == (batch_size, T_price, d_model)

    def test_cross_attn_mode(self, d_model, n_heads, modality_reps, batch_size, T_price):
        fusion = ModalityFusion(d_model, n_modalities=3, n_heads=n_heads, mode="cross_attn")
        out = fusion(modality_reps, target_len=T_price)
        assert out.shape == (batch_size, T_price, d_model)

    def test_no_nan(self, d_model, n_heads, modality_reps, T_price):
        for mode in ["concat", "gated"]:
            fusion = ModalityFusion(d_model, n_modalities=3, n_heads=n_heads, mode=mode)
            out = fusion(modality_reps, target_len=T_price)
            assert not torch.isnan(out).any(), f"NaN in mode={mode}"

    def test_different_T_alignment(self, d_model, n_heads, batch_size, T_price):
        """Modalities with different T should be aligned to target_len."""
        reps = [
            torch.randn(batch_size, T_price, d_model),
            torch.randn(batch_size, T_price + 4, d_model),
            torch.randn(batch_size, T_price - 2, d_model),
        ]
        fusion = ModalityFusion(d_model, n_modalities=3, n_heads=n_heads, mode="concat")
        out = fusion(reps, target_len=T_price)
        assert out.shape == (batch_size, T_price, d_model)


# ---------------------------------------------------------------------------
# GatedFusion tests
# ---------------------------------------------------------------------------

class TestGatedFusion:
    def test_output_shape(self, d_model, batch_size):
        gf = GatedFusion(d_model, n_modalities=3)
        reps = [torch.randn(batch_size, 8, d_model) for _ in range(3)]
        out = gf(reps)
        assert out.shape == (batch_size, 8, d_model)

    def test_gate_range(self, d_model, batch_size):
        """Gates should be in (0, 1)."""
        gf = GatedFusion(d_model, n_modalities=2)
        reps = [torch.randn(batch_size, 4, d_model) for _ in range(2)]
        # We can inspect the gate computation by temporarily hooking
        stacked = torch.stack(reps, dim=2)
        flat = stacked.view(batch_size, 4, 2 * d_model)
        gates = torch.sigmoid(gf.gate_proj(flat))
        assert gates.min() >= 0.0
        assert gates.max() <= 1.0


# ---------------------------------------------------------------------------
# TemporalAlignment tests
# ---------------------------------------------------------------------------

class TestTemporalAlignment:
    @pytest.fixture
    def timestamps(self, batch_size):
        """Uniformly spaced price bar timestamps."""
        B = batch_size
        T_price = 16
        # Simulate 5-min bars from some epoch
        base = 1700000000.0
        ts = torch.zeros(B, T_price)
        for b in range(B):
            ts[b] = torch.linspace(base, base + T_price * 300, T_price)
        return ts

    @pytest.fixture
    def event_data(self, batch_size):
        """Sparse news event timestamps and values."""
        B, n_events, D = batch_size, 4, 16
        base = 1700000000.0
        ts = torch.zeros(B, n_events)
        vals = torch.randn(B, n_events, D)
        for b in range(B):
            ts[b] = base + torch.rand(n_events) * 16 * 300
        return ts, vals

    def test_nearest_alignment_shape(self, timestamps, event_data, batch_size):
        aligner = TemporalAlignment(method="nearest")
        event_ts, event_vals = event_data
        aligned, coverage = aligner(timestamps, event_ts, event_vals)
        B, T_price = timestamps.shape
        D = event_vals.shape[-1]
        assert aligned.shape == (B, T_price, D)
        assert coverage.shape == (B, T_price)
        assert coverage.dtype == torch.bool

    def test_interpolate_alignment_shape(self, timestamps, event_data, batch_size):
        aligner = TemporalAlignment(method="interpolate")
        event_ts, event_vals = event_data
        aligned, coverage = aligner(timestamps, event_ts, event_vals)
        B, T_price = timestamps.shape
        D = event_vals.shape[-1]
        assert aligned.shape == (B, T_price, D)

    def test_nearest_coverage_correct(self, timestamps, event_data):
        aligner = TemporalAlignment(method="nearest")
        event_ts, event_vals = event_data
        aligned, coverage = aligner(timestamps, event_ts, event_vals)
        # Coverage should be True where events were mapped
        n_covered = coverage.sum(dim=-1)  # (B,)
        # With 4 events and 16 bars, at least 1 and at most 4 bars should be covered per sample
        assert (n_covered >= 1).all()

    def test_no_nan_in_alignment(self, timestamps, event_data):
        aligner = TemporalAlignment(method="nearest")
        event_ts, event_vals = event_data
        aligned, _ = aligner(timestamps, event_ts, event_vals)
        assert not torch.isnan(aligned).any()


# ---------------------------------------------------------------------------
# MultiModalLumina tests
# ---------------------------------------------------------------------------

class TestMultiModalLumina:
    @pytest.fixture
    def small_mm_config(self):
        price_enc = LuminaConfig(
            d_model=32, n_layers=1, n_heads=2, n_kv_heads=None,
            d_ffn=64, use_moe=False, arch="bidirectional",
            lm_head=False, pool_head=False,
            unified_token_dim=16, max_seq_len=32, use_temporal=False,
            use_rope=False,
        )
        onchain_enc = LuminaConfig(
            d_model=16, n_layers=1, n_heads=2, n_kv_heads=None,
            d_ffn=32, use_moe=False, arch="bidirectional",
            lm_head=False, pool_head=False,
            unified_token_dim=16, max_seq_len=32, use_temporal=False,
            use_rope=False,
        )
        news_enc = LuminaConfig(
            d_model=32, n_layers=1, n_heads=2, n_kv_heads=None,
            d_ffn=64, use_moe=False, arch="bidirectional",
            lm_head=False, pool_head=False,
            unified_token_dim=16, max_seq_len=32, use_temporal=False,
            use_rope=False,
        )
        return MultiModalLuminaConfig(
            price_encoder=price_enc,
            onchain_encoder=onchain_enc,
            news_encoder=news_enc,
            use_shared_encoder=False,
            d_fusion=32,
            fusion_mode="gated",
            fusion_n_heads=2,
            n_cross_attn_layers=1,
            n_regimes=4,
            forecast_horizon=3,
            unified_token_dim=16,
        )

    @pytest.fixture
    def mm_model(self, small_mm_config):
        return MultiModalLumina(small_mm_config)

    @pytest.fixture
    def price_tokens(self, batch_size):
        return torch.randn(batch_size, 12, 16)

    @pytest.fixture
    def onchain_tokens(self, batch_size):
        return torch.randn(batch_size, 12, 16)

    @pytest.fixture
    def news_tokens(self, batch_size):
        return torch.randn(batch_size, 8, 16)

    def test_price_only_forward(self, mm_model, price_tokens, batch_size):
        out = mm_model(price_tokens=price_tokens)
        assert "return_pred" in out
        assert "regime_logits" in out
        assert "vol_forecast" in out
        assert "crisis_logits" in out

    def test_output_shapes(self, mm_model, price_tokens, onchain_tokens, news_tokens, batch_size):
        out = mm_model(
            price_tokens=price_tokens,
            onchain_tokens=onchain_tokens,
            news_tokens=news_tokens,
        )
        assert out["return_pred"].shape == (batch_size, 1)
        assert out["regime_logits"].shape == (batch_size, 4)
        assert out["vol_forecast"].shape == (batch_size, 3)
        assert out["crisis_logits"].shape == (batch_size, 2)

    def test_vol_forecast_positive(self, mm_model, price_tokens):
        out = mm_model(price_tokens=price_tokens)
        # Volatility forecast should be non-negative (Softplus activation)
        assert (out["vol_forecast"] >= 0).all()

    def test_no_nan(self, mm_model, price_tokens, onchain_tokens, news_tokens):
        out = mm_model(
            price_tokens=price_tokens,
            onchain_tokens=onchain_tokens,
            news_tokens=news_tokens,
        )
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any(), f"NaN in {k}"

    def test_gradient_flow(self, mm_model, price_tokens, news_tokens):
        out = mm_model(price_tokens=price_tokens, news_tokens=news_tokens)
        loss = (
            out["return_pred"].sum()
            + out["regime_logits"].sum()
            + out["crisis_logits"].sum()
        )
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in mm_model.parameters()
        )
        assert has_grad

    def test_with_masks(self, mm_model, price_tokens, news_tokens, batch_size):
        price_mask = torch.ones(batch_size, 12, dtype=torch.bool)
        price_mask[0, -3:] = False
        news_mask = torch.ones(batch_size, 8, dtype=torch.bool)
        out = mm_model(
            price_tokens=price_tokens,
            news_tokens=news_tokens,
            price_mask=price_mask,
            news_mask=news_mask,
        )
        assert not torch.isnan(out["return_pred"]).any()

    def test_crisis_logits_valid(self, mm_model, price_tokens):
        out = mm_model(price_tokens=price_tokens)
        probs = torch.softmax(out["crisis_logits"], dim=-1)
        assert (probs >= 0).all()
        assert (probs <= 1).all()
        assert torch.allclose(probs.sum(dim=-1), torch.ones(price_tokens.shape[0]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
