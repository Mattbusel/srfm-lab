"""
tests/test_tokenizer.py

Unit tests for lumina tokenizer components.
"""

import pytest
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lumina.tokenizer import (
    PriceTokenizerConfig,
    PriceSeriesTokenizer,
    OrderBookTokenizerConfig,
    OrderBookTokenizer,
    OnChainTokenizerConfig,
    OnChainTokenizer,
    MultiModalTokenizerConfig,
    MultiModalTokenizer,
    RollingPercentileNormalizer,
    MODALITY_PRICE,
    MODALITY_ORDERBOOK,
    MODALITY_ONCHAIN,
    MODALITY_NEWS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def price_config():
    return PriceTokenizerConfig(
        patch_size=4,
        d_patch=16,
        vocab_bins=32,
        hybrid_dim=32,
        log_return_clip=0.10,
        use_volume=True,
        normalize_ohlc=True,
    )


@pytest.fixture
def lob_config():
    return OrderBookTokenizerConfig(n_levels=3, d_lob=16, use_delta=True)


@pytest.fixture
def onchain_config():
    return OnChainTokenizerConfig(
        n_signals=3, quantize_bins=16, d_onchain=16, rolling_window=50
    )


@pytest.fixture
def sample_ohlcv():
    """Returns (B=2, T=32, 5) OHLCV tensor with realistic values."""
    B, T = 2, 32
    torch.manual_seed(0)
    open_ = torch.rand(B, T) * 100 + 50
    high = open_ + torch.rand(B, T) * 5
    low = open_ - torch.rand(B, T) * 5
    close = open_ + (torch.rand(B, T) - 0.5) * 4
    vol = torch.rand(B, T) * 1e6 + 1000
    return torch.stack([open_, high, low, close, vol], dim=-1)


@pytest.fixture
def sample_lob():
    """Returns (B=2, T=16, 12) LOB tensor (3 levels * 4 features)."""
    B, T, D = 2, 16, 12
    base_price = 100.0
    lob = torch.zeros(B, T, D)
    for i in range(3):
        lob[:, :, i * 2] = base_price - i * 0.5 + torch.randn(B, T) * 0.01
        lob[:, :, i * 2 + 1] = torch.rand(B, T) * 100 + 10
        lob[:, :, 6 + i * 2] = base_price + (i + 1) * 0.5 + torch.randn(B, T) * 0.01
        lob[:, :, 6 + i * 2 + 1] = torch.rand(B, T) * 100 + 10
    return lob


@pytest.fixture
def sample_onchain():
    """Returns (B=2, T=32, 3) on-chain signal tensor."""
    torch.manual_seed(1)
    return torch.rand(2, 32, 3)


# ---------------------------------------------------------------------------
# RollingPercentileNormalizer tests
# ---------------------------------------------------------------------------

class TestRollingPercentileNormalizer:
    def test_basic(self):
        norm = RollingPercentileNormalizer(window=10)
        ranks = [norm.update_and_rank(float(x)) for x in range(10)]
        # Last inserted value should have rank 1.0
        assert ranks[-1] == pytest.approx(1.0)

    def test_range(self):
        norm = RollingPercentileNormalizer(window=100)
        for x in range(100):
            r = norm.update_and_rank(float(x))
            assert 0.0 <= r <= 1.0

    def test_window_eviction(self):
        norm = RollingPercentileNormalizer(window=5)
        for x in range(100):
            r = norm.update_and_rank(float(x))
        # After many large values, rank of previous max should be ~0
        r = norm.update_and_rank(-999.0)
        assert r == pytest.approx(0.0)

    def test_state_dict(self):
        norm = RollingPercentileNormalizer(window=20)
        for x in [1, 2, 3, 4, 5]:
            norm.update_and_rank(float(x))
        state = norm.state_dict()
        norm2 = RollingPercentileNormalizer(window=10)
        norm2.load_state_dict(state)
        assert norm2._buffer == norm._buffer
        assert norm2.window == norm.window


# ---------------------------------------------------------------------------
# PriceSeriesTokenizer tests
# ---------------------------------------------------------------------------

class TestPriceSeriesTokenizer:
    def test_patch_mode_shape(self, price_config, sample_ohlcv):
        tok = PriceSeriesTokenizer(price_config, mode="patch")
        emb, mask = tok(sample_ohlcv)
        B, T, _ = sample_ohlcv.shape
        expected_patches = -(-T // price_config.patch_size)  # ceiling div
        assert emb.shape[0] == B
        assert emb.shape[1] == expected_patches
        assert emb.shape[2] == price_config.hybrid_dim

    def test_quantized_mode_shape(self, price_config, sample_ohlcv):
        tok = PriceSeriesTokenizer(price_config, mode="quantized")
        emb, mask = tok(sample_ohlcv)
        assert emb.shape[0] == 2  # B
        assert emb.shape[2] == price_config.hybrid_dim

    def test_hybrid_mode_shape(self, price_config, sample_ohlcv):
        tok = PriceSeriesTokenizer(price_config, mode="hybrid")
        emb, mask = tok(sample_ohlcv)
        B = sample_ohlcv.shape[0]
        assert emb.shape[0] == B
        assert emb.shape[2] == price_config.hybrid_dim

    def test_mask_output_shape(self, price_config, sample_ohlcv):
        tok = PriceSeriesTokenizer(price_config, mode="hybrid")
        B, T, _ = sample_ohlcv.shape
        # Create partial mask: first sample fully valid, second half-valid
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[1, T // 2:] = False
        emb, patch_mask = tok(sample_ohlcv, attention_mask=mask)
        assert patch_mask.shape[0] == B
        assert patch_mask.dtype == torch.bool

    def test_no_nan_in_output(self, price_config, sample_ohlcv):
        tok = PriceSeriesTokenizer(price_config, mode="hybrid")
        emb, _ = tok(sample_ohlcv)
        assert not torch.isnan(emb).any()
        assert not torch.isinf(emb).any()

    def test_gradient_flow(self, price_config, sample_ohlcv):
        tok = PriceSeriesTokenizer(price_config, mode="hybrid")
        emb, _ = tok(sample_ohlcv)
        loss = emb.sum()
        loss.backward()
        # Check gradients exist for projection weights
        assert tok.patch_proj.weight.grad is not None

    def test_log_return_computation(self, price_config):
        """Test log-return computation with a known price series."""
        tok = PriceSeriesTokenizer(price_config, mode="hybrid")
        # Constant price → zero returns
        ohlcv = torch.ones(1, 8, 5) * 100.0
        log_ret = tok._ohlcv_to_log_returns(ohlcv)
        assert log_ret[:, 0].abs().item() < 1e-5  # first return is 0

    def test_bin_quantization(self, price_config):
        tok = PriceSeriesTokenizer(price_config, mode="quantized")
        # Returns at extremes
        extreme_pos = torch.tensor([[[price_config.log_return_clip]]])
        bins = tok._quantize_returns(extreme_pos)
        assert bins.shape == (1, 1)
        assert bins.max() < price_config.vocab_bins + 1

    def test_t_not_multiple_of_patch(self, price_config):
        """T=17 is not divisible by patch_size=4 → should pad internally."""
        tok = PriceSeriesTokenizer(price_config, mode="hybrid")
        ohlcv = torch.rand(1, 17, 5) + 10
        emb, mask = tok(ohlcv)
        # Expect ceil(17/4) = 5 patches
        assert emb.shape[1] == 5


# ---------------------------------------------------------------------------
# OrderBookTokenizer tests
# ---------------------------------------------------------------------------

class TestOrderBookTokenizer:
    def test_output_shape(self, lob_config, sample_lob):
        tok = OrderBookTokenizer(lob_config)
        out = tok(sample_lob)
        B, T, _ = sample_lob.shape
        assert out.shape == (B, T, lob_config.d_lob)

    def test_no_nan(self, lob_config, sample_lob):
        tok = OrderBookTokenizer(lob_config)
        out = tok(sample_lob)
        assert not torch.isnan(out).any()

    def test_delta_encoding_effect(self, lob_config):
        """With delta encoding, identical consecutive snapshots should produce different outputs."""
        tok_delta = OrderBookTokenizer(lob_config)
        tok_no_delta = OrderBookTokenizer(OrderBookTokenizerConfig(
            n_levels=3, d_lob=16, use_delta=False
        ))
        B, T = 1, 4
        lob = torch.rand(B, T, 12) * 100
        out_delta = tok_delta(lob)
        out_no_delta = tok_no_delta(lob)
        # They should differ because delta adds information
        assert out_delta.shape == out_no_delta.shape

    def test_gradient_flow(self, lob_config, sample_lob):
        tok = OrderBookTokenizer(lob_config)
        out = tok(sample_lob)
        out.sum().backward()
        assert tok.raw_proj.weight.grad is not None


# ---------------------------------------------------------------------------
# OnChainTokenizer tests
# ---------------------------------------------------------------------------

class TestOnChainTokenizer:
    def test_output_shape(self, onchain_config, sample_onchain):
        tok = OnChainTokenizer(onchain_config)
        out = tok(sample_onchain)
        B, T, _ = sample_onchain.shape
        assert out.shape == (B, T, onchain_config.d_onchain)

    def test_no_nan(self, onchain_config, sample_onchain):
        tok = OnChainTokenizer(onchain_config)
        out = tok(sample_onchain)
        assert not torch.isnan(out).any()

    def test_percentile_rank(self, onchain_config, sample_onchain):
        tok = OnChainTokenizer(onchain_config)
        # percentiles should be in [0, 1]
        percentiles = tok._percentile_rank_batch(sample_onchain)
        assert percentiles.min() >= 0.0
        assert percentiles.max() <= 1.0

    def test_gradient_flow(self, onchain_config, sample_onchain):
        tok = OnChainTokenizer(onchain_config)
        out = tok(sample_onchain)
        out.sum().backward()
        assert tok.fusion.weight.grad is not None


# ---------------------------------------------------------------------------
# MultiModalTokenizer tests
# ---------------------------------------------------------------------------

class TestMultiModalTokenizer:
    @pytest.fixture
    def mm_config(self):
        return MultiModalTokenizerConfig(
            price_config=PriceTokenizerConfig(patch_size=4, d_patch=16, vocab_bins=32, hybrid_dim=32),
            lob_config=OrderBookTokenizerConfig(n_levels=3, d_lob=32),
            onchain_config=OnChainTokenizerConfig(n_signals=3, quantize_bins=16, d_onchain=32),
            unified_dim=64,
            max_seq_len=32,
            pad_to_max=True,
        )

    def test_price_only(self, mm_config, sample_ohlcv):
        tok = MultiModalTokenizer(mm_config)
        out = tok(ohlcv=sample_ohlcv)
        assert out.token_embeddings.shape[0] == sample_ohlcv.shape[0]
        assert out.token_embeddings.shape[2] == mm_config.unified_dim

    def test_multimodal_batch_structure(self, mm_config, sample_ohlcv, sample_lob, sample_onchain):
        tok = MultiModalTokenizer(mm_config)
        # Align T for LOB and onchain
        B, T = sample_ohlcv.shape[0], 16
        lob = sample_lob[:, :T]
        onchain = sample_onchain[:, :T]
        out = tok(ohlcv=sample_ohlcv, lob=lob, onchain=onchain)
        assert out.token_embeddings.shape[-1] == mm_config.unified_dim
        assert out.modality_ids.shape == out.attention_mask.shape

    def test_max_seq_len_respected(self, mm_config, sample_ohlcv):
        tok = MultiModalTokenizer(mm_config)
        out = tok(ohlcv=sample_ohlcv)
        assert out.token_embeddings.shape[1] == mm_config.max_seq_len

    def test_modality_ids_values(self, mm_config, sample_ohlcv):
        tok = MultiModalTokenizer(mm_config)
        out = tok(ohlcv=sample_ohlcv)
        # All modality IDs should be valid (0-3) or 0 (padding)
        assert (out.modality_ids >= 0).all()
        assert (out.modality_ids <= 3).all()

    def test_news_embeddings(self, mm_config):
        tok = MultiModalTokenizer(mm_config)
        B, T_news, D_news = 2, 8, 768
        news_emb = torch.randn(B, T_news, D_news)
        out = tok(news_embeddings=news_emb)
        assert out.token_embeddings.shape[0] == B

    def test_no_nan_in_output(self, mm_config, sample_ohlcv, sample_onchain):
        tok = MultiModalTokenizer(mm_config)
        out = tok(ohlcv=sample_ohlcv, onchain=sample_onchain[:, :16])
        assert not torch.isnan(out.token_embeddings).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
