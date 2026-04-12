"""Tests for additional Lumina model variants."""
import pytest
import torch

class TestLuminaNano:
    def test_forward_shape(self):
        from model import LuminaNano
        model = LuminaNano(input_dim=8, d_model=16, num_heads=2, num_layers=2, num_classes=3)
        x = torch.randn(2, 16, 8)
        out = model(x)
        assert out.shape == (2, 3)

    def test_no_nan(self):
        from model import LuminaNano
        model = LuminaNano()
        x = torch.randn(4, 32, 8)
        assert not torch.isnan(model(x)).any()

    def test_gradient(self):
        from model import LuminaNano
        model = LuminaNano(input_dim=8, d_model=16, num_heads=2, num_layers=2)
        x = torch.randn(2, 8, 8, requires_grad=True)
        model(x).sum().backward()
        assert x.grad is not None

class TestLuminaAlpha:
    def test_forward_returns_dict(self):
        from model import LuminaAlpha
        model = LuminaAlpha(input_dim=16, d_model=32, num_heads=4, num_layers=2,
                             horizons=[1, 5, 21], max_seq_len=64)
        x = torch.randn(2, 20, 16)
        out = model(x)
        assert 'pred_1d' in out
        assert 'pred_5d' in out
        assert 'pred_21d' in out

    def test_pred_shapes(self):
        from model import LuminaAlpha
        model = LuminaAlpha(input_dim=8, d_model=16, num_heads=2, num_layers=2,
                             horizons=[1, 5], max_seq_len=32)
        x = torch.randn(4, 10, 8)
        out = model(x)
        assert out['pred_1d'].shape == (4,)
        assert out['logvar_1d'].shape == (4,)

    def test_no_nan(self):
        from model import LuminaAlpha
        model = LuminaAlpha(input_dim=8, d_model=16, num_heads=2, num_layers=2,
                             horizons=[1], max_seq_len=32)
        x = torch.randn(2, 8, 8)
        out = model(x)
        assert not torch.isnan(out['pred_1d']).any()

class TestLuminaRiskModel:
    def test_forward_returns_dict(self):
        from model import LuminaRiskModel
        model = LuminaRiskModel(n_assets=50, n_factors=5, d_model=32, num_heads=4, num_layers=2)
        asset_ids = torch.randint(0, 50, (2, 10))
        out = model(asset_ids)
        assert 'beta' in out
        assert 'cov_matrix' in out

    def test_cov_matrix_shape(self):
        from model import LuminaRiskModel
        model = LuminaRiskModel(n_assets=20, n_factors=4, d_model=16, num_heads=2, num_layers=2)
        ids = torch.randint(0, 20, (2, 5))
        out = model(ids)
        assert out['cov_matrix'].shape == (2, 5, 5)

    def test_no_nan(self):
        from model import LuminaRiskModel
        model = LuminaRiskModel(n_assets=20, n_factors=4, d_model=16, num_heads=2, num_layers=2)
        ids = torch.randint(0, 20, (2, 5))
        out = model(ids)
        assert not torch.isnan(out['cov_matrix']).any()

class TestLuminaSentimentModel:
    def test_forward_returns_dict(self):
        from model import LuminaSentimentModel
        model = LuminaSentimentModel(text_embed_dim=32, market_dim=8, d_model=16,
                                      num_heads=2, num_layers=2)
        text = torch.randn(2, 32)
        out = model(text)
        assert 'sentiment' in out
        assert out['sentiment'].shape == (2,)

    def test_with_market_features(self):
        from model import LuminaSentimentModel
        model = LuminaSentimentModel(text_embed_dim=32, market_dim=8, d_model=16,
                                      num_heads=2, num_layers=2)
        text = torch.randn(2, 32)
        market = torch.randn(2, 8)
        out = model(text, market)
        assert out['sentiment'].shape == (2,)
        assert not torch.isnan(out['sentiment']).any()

class TestLuminaTrendFollower:
    def test_forward_returns_dict(self):
        from model import LuminaTrendFollower
        model = LuminaTrendFollower(input_dim=8, d_model=16, num_heads=2, num_layers=2, seq_len=20)
        x = torch.randn(2, 20, 8)
        out = model(x)
        assert 'regime_logits' in out
        assert 'position' in out
        assert out['regime_logits'].shape == (2, 4)

    def test_position_bounded(self):
        from model import LuminaTrendFollower
        model = LuminaTrendFollower(input_dim=8, d_model=16, num_heads=2, num_layers=2, seq_len=16)
        x = torch.randn(4, 16, 8)
        out = model(x)
        assert out['position'].abs().max() <= 1.0 + 1e-5

class TestLuminaMarketMaker:
    def test_forward_shape(self):
        from model import LuminaMarketMaker
        model = LuminaMarketMaker(orderbook_levels=5, d_model=16, num_heads=2, num_layers=2)
        ob = torch.randn(2, 10, 20)  # 4*5=20 features
        out = model(ob)
        assert 'bid_spread_bps' in out
        assert out['bid_spread_bps'].shape == (2,)

    def test_spreads_positive(self):
        from model import LuminaMarketMaker
        model = LuminaMarketMaker(orderbook_levels=5, d_model=16, num_heads=2, num_layers=2)
        ob = torch.randn(4, 8, 20)
        out = model(ob)
        assert (out['bid_spread_bps'] > 0).all()
        assert (out['ask_spread_bps'] > 0).all()

@pytest.mark.parametrize('input_dim,d_model,num_heads,B,T,num_classes', [
    (4, 16, 2, 1, 8, 2),
    (4, 16, 2, 1, 8, 3),
    (4, 16, 2, 1, 16, 2),
    (4, 16, 2, 1, 16, 3),
    (4, 16, 2, 2, 8, 2),
    (4, 16, 2, 2, 8, 3),
    (4, 16, 2, 2, 16, 2),
    (4, 16, 2, 2, 16, 3),
    (4, 16, 4, 1, 8, 2),
    (4, 16, 4, 1, 8, 3),
    (4, 16, 4, 1, 16, 2),
    (4, 16, 4, 1, 16, 3),
    (4, 16, 4, 2, 8, 2),
    (4, 16, 4, 2, 8, 3),
    (4, 16, 4, 2, 16, 2),
    (4, 16, 4, 2, 16, 3),
    (4, 32, 2, 1, 8, 2),
    (4, 32, 2, 1, 8, 3),
    (4, 32, 2, 1, 16, 2),
    (4, 32, 2, 1, 16, 3),
    (4, 32, 2, 2, 8, 2),
    (4, 32, 2, 2, 8, 3),
    (4, 32, 2, 2, 16, 2),
    (4, 32, 2, 2, 16, 3),
    (4, 32, 4, 1, 8, 2),
    (4, 32, 4, 1, 8, 3),
    (4, 32, 4, 1, 16, 2),
    (4, 32, 4, 1, 16, 3),
    (4, 32, 4, 2, 8, 2),
    (4, 32, 4, 2, 8, 3),
    (4, 32, 4, 2, 16, 2),
    (4, 32, 4, 2, 16, 3),
    (8, 16, 2, 1, 8, 2),
    (8, 16, 2, 1, 8, 3),
    (8, 16, 2, 1, 16, 2),
    (8, 16, 2, 1, 16, 3),
    (8, 16, 2, 2, 8, 2),
    (8, 16, 2, 2, 8, 3),
    (8, 16, 2, 2, 16, 2),
    (8, 16, 2, 2, 16, 3),
    (8, 16, 4, 1, 8, 2),
    (8, 16, 4, 1, 8, 3),
    (8, 16, 4, 1, 16, 2),
    (8, 16, 4, 1, 16, 3),
    (8, 16, 4, 2, 8, 2),
    (8, 16, 4, 2, 8, 3),
    (8, 16, 4, 2, 16, 2),
    (8, 16, 4, 2, 16, 3),
    (8, 32, 2, 1, 8, 2),
    (8, 32, 2, 1, 8, 3),
    (8, 32, 2, 1, 16, 2),
    (8, 32, 2, 1, 16, 3),
    (8, 32, 2, 2, 8, 2),
    (8, 32, 2, 2, 8, 3),
    (8, 32, 2, 2, 16, 2),
    (8, 32, 2, 2, 16, 3),
    (8, 32, 4, 1, 8, 2),
    (8, 32, 4, 1, 8, 3),
    (8, 32, 4, 1, 16, 2),
    (8, 32, 4, 1, 16, 3),
    (8, 32, 4, 2, 8, 2),
    (8, 32, 4, 2, 8, 3),
    (8, 32, 4, 2, 16, 2),
    (8, 32, 4, 2, 16, 3),
    (16, 16, 2, 1, 8, 2),
    (16, 16, 2, 1, 8, 3),
    (16, 16, 2, 1, 16, 2),
    (16, 16, 2, 1, 16, 3),
    (16, 16, 2, 2, 8, 2),
    (16, 16, 2, 2, 8, 3),
    (16, 16, 2, 2, 16, 2),
    (16, 16, 2, 2, 16, 3),
    (16, 16, 4, 1, 8, 2),
    (16, 16, 4, 1, 8, 3),
    (16, 16, 4, 1, 16, 2),
    (16, 16, 4, 1, 16, 3),
    (16, 16, 4, 2, 8, 2),
    (16, 16, 4, 2, 8, 3),
    (16, 16, 4, 2, 16, 2),
    (16, 16, 4, 2, 16, 3),
    (16, 32, 2, 1, 8, 2),
    (16, 32, 2, 1, 8, 3),
    (16, 32, 2, 1, 16, 2),
    (16, 32, 2, 1, 16, 3),
    (16, 32, 2, 2, 8, 2),
    (16, 32, 2, 2, 8, 3),
    (16, 32, 2, 2, 16, 2),
    (16, 32, 2, 2, 16, 3),
    (16, 32, 4, 1, 8, 2),
    (16, 32, 4, 1, 8, 3),
    (16, 32, 4, 1, 16, 2),
    (16, 32, 4, 1, 16, 3),
    (16, 32, 4, 2, 8, 2),
    (16, 32, 4, 2, 8, 3),
    (16, 32, 4, 2, 16, 2),
    (16, 32, 4, 2, 16, 3),
])
def test_lumina_nano_parametrized(input_dim, d_model, num_heads, B, T, num_classes):
    from model import LuminaNano
    model = LuminaNano(input_dim=input_dim, d_model=d_model, num_heads=num_heads,
                        num_layers=2, max_seq_len=T+4, num_classes=num_classes)
    x = torch.randn(B, T, input_dim)
    out = model(x)
    assert out.shape == (B, num_classes)
    assert not torch.isnan(out).any()
